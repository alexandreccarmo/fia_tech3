"""
[REQ-2][REQ-3c] Chain de resposta ancorada em fontes.

O QUE FAZ:
    Monta o prompt com o contexto recuperado e o quadro clínico do paciente,
    invoca a LLM customizada e devolve o texto gerado.

O PADRÃO É O DAS AULAS:
    `prompt | llm | parser`, com `.invoke()`. A diferença em relação ao
    exemplo do curso está no que entra no prompt: em vez de um texto solto, um
    bloco de fontes já marcadas com [E#], [P#] e [C#], montado pelo
    recuperador. É esse detalhe que transforma "responder com contexto" em
    "responder com fonte rastreável".

A ORDEM DOS BLOCOS NO PROMPT É DELIBERADA:
    1. quadro clínico do paciente (quando houver)
    2. fontes recuperadas
    3. pergunta
    4. instruções de correção (só na reescrita)

    O paciente vem primeiro porque é o filtro que deve condicionar a leitura
    de tudo o que vem depois: o modelo precisa saber que há alergia a
    betalactâmico ANTES de ler o protocolo que recomenda ceftriaxona. Colocar
    o prontuário no fim faria o modelo formar a resposta e só então encontrar
    a contraindicação.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from medgraph.auditoria import TipoEvento, registrar
from medgraph.chains import prompts
from medgraph.logging_config import obter_logger
from medgraph.rag.recuperador import Trecho

log = obter_logger(__name__)


def montar_mensagem_usuario(
    pergunta: str,
    trechos: Sequence[Trecho],
    *,
    contexto_paciente: str = "",
    intencao: str = "",
    instrucoes_correcao: str = "",
    aviso_emergencia: str = "",
) -> str:
    """Monta o conteúdo da mensagem do usuário, na ordem descrita no cabeçalho."""
    blocos: list[str] = []

    if aviso_emergencia:
        blocos.append(f"[ATENÇÃO]\n{aviso_emergencia}")

    if contexto_paciente:
        blocos.append(
            "QUADRO CLÍNICO DO PACIENTE (marcador de citação: [C1])\n" + contexto_paciente
        )

    if trechos:
        blocos.append(
            "FONTES DISPONÍVEIS\n"
            + prompts.montar_contexto([t.para_prompt() for t in trechos])
        )
    else:
        blocos.append(
            "FONTES DISPONÍVEIS\n(nenhuma fonte relevante foi recuperada para esta pergunta)"
        )

    instrucao_tarefa = {
        "conduta_terapeutica": (
            "Apresente o que os protocolos internos e a evidência dizem sobre as opções "
            "terapêuticas. NÃO prescreva. Aponte explicitamente qualquer conflito com "
            "alergias, medicações em uso ou exames do paciente."
        ),
        "exames_pendentes": (
            "Informe quais exames estão pendentes e o que cada um agrega à condução do caso."
        ),
        "resumo_prontuario": (
            "Sintetize o quadro do paciente em tópicos, destacando alergias, medicações "
            "ativas, exames críticos e pendências."
        ),
        "consulta_paciente": "Responda objetivamente com base nos dados do prontuário.",
    }.get(intencao, prompts.INSTRUCAO_PROTOCOLO)

    blocos.append(f"TAREFA\n{instrucao_tarefa}")
    blocos.append(f"PERGUNTA DO MÉDICO\n{pergunta}")

    if instrucoes_correcao:
        blocos.append(f"CORREÇÃO OBRIGATÓRIA\n{instrucoes_correcao}")

    return "\n\n".join(blocos)


def responder(
    llm,
    pergunta: str,
    trechos: Sequence[Trecho],
    *,
    contexto_paciente: str = "",
    intencao: str = "",
    instrucoes_correcao: str = "",
    aviso_emergencia: str = "",
) -> str:
    """
    Gera a resposta ancorada nas fontes.

    A composição `mensagens | llm | StrOutputParser()` é o pipeline moderno do
    LangChain — o mesmo `prompt | llm` das aulas, com o parser garantindo que
    a saída chegue como texto puro em vez de objeto de mensagem.
    """
    mensagem = montar_mensagem_usuario(
        pergunta,
        trechos,
        contexto_paciente=contexto_paciente,
        intencao=intencao,
        instrucoes_correcao=instrucoes_correcao,
        aviso_emergencia=aviso_emergencia,
    )

    cadeia = llm | StrOutputParser()
    resposta = cadeia.invoke(
        [SystemMessage(content=prompts.SISTEMA), HumanMessage(content=mensagem)]
    )

    registrar(
        TipoEvento.LLM,
        "Resposta gerada",
        etapa="raciocinio_clinico",
        intencao=intencao,
        fontes_no_prompt=[t.marcador for t in trechos],
        tem_contexto_paciente=bool(contexto_paciente),
        e_reescrita=bool(instrucoes_correcao),
        caracteres_prompt=len(mensagem),
        caracteres_resposta=len(resposta),
    )
    return resposta.strip()
