"""
[REQ-2] Chain de triagem — classificação da intenção.

O QUE FAZ:
    Decide, a partir da pergunta do médico, qual dos cinco caminhos do fluxo
    deve ser seguido: dúvida clínica, consulta ao paciente, exames pendentes,
    conduta terapêutica ou resumo de prontuário.

POR QUE ISSO É UMA ROUTER CHAIN, NO SENTIDO DA AULA 04:
    É exatamente o padrão de roteamento visto no curso — um prompt classifica
    a entrada e o resultado seleciona a cadeia seguinte. A diferença é que
    aqui o destino não é outra chain, e sim uma aresta condicional do
    LangGraph, o que permite ao fluxo ter estado, ciclos e um ponto de
    validação humana.

POR QUE O ROTEAMENTO IMPORTA PARA A SEGURANÇA, E NÃO SÓ PARA A EFICIÊNCIA:
    A intenção `conduta_terapeutica` é a única que pode gerar texto parecido
    com prescrição, e por isso está marcada em `politicas.yaml` com
    `sempre_validacao_humana: true`. Classificar corretamente é o que faz o
    fluxo saber que precisa parar e pedir validação. Uma classificação errada
    para baixo — tratar uma pergunta de conduta como dúvida conceitual — pula
    a validação. É o erro mais caro possível aqui.

    Daí a REGRA DE DESEMPATE conservadora do prompt: na dúvida entre duas
    intenções, escolher a de maior exigência de validação.

ESTRATÉGIA EM DOIS NÍVEIS:
    1. Heurística por palavra-chave, determinística e instantânea.
    2. LLM, apenas quando a heurística não é conclusiva.

    Não é economia de tokens: é previsibilidade. "Quais exames estão
    pendentes do PAC-0012?" não precisa de um modelo para ser classificada, e
    delegá-la ao modelo introduziria variação onde não há ambiguidade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from medgraph.auditoria import TipoEvento, registrar
from medgraph.guardrails import politicas as mod_politicas
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

INTENCAO_PADRAO = "duvida_clinica"

# Padrões que identificam a intenção sem ambiguidade. A ordem importa: são
# avaliados de cima para baixo, e o primeiro que casar decide.
#
# `conduta_terapeutica` vem primeiro de propósito. É a intenção de maior
# risco, e o custo de classificá-la a menos (pular a validação humana) é
# muito maior do que o de classificá-la a mais (pedir uma validação
# desnecessária).
HEURISTICAS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "conduta_terapeutica",
        re.compile(
            r"\b(qual|que)\s+(conduta|tratamento|antibi[óo]tico|esquema|terap[êe]utica)"
            r"|\b(como|devo)\s+(tratar|conduzir|manejar|iniciar)"
            r"|\b(posso|devo)\s+(prescrever|iniciar|associar|suspender|ajustar)"
            r"|\bajust(ar|e)\s+(a\s+)?dose"
            r"|\bo que fa[çc]o\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exames_pendentes",
        re.compile(
            r"\bexames?\s+(pendentes?|em aberto|sem resultado|aguardando)"
            r"|\b(o que|quais|algo)\s+(est[áa]|ainda)\s+pendente"
            r"|\bpendências?\s+(de\s+)?exames?",
            re.IGNORECASE,
        ),
    ),
    (
        "resumo_prontuario",
        re.compile(
            r"\b(resum[ao]|res[úu]me|sintetiz|panorama|hist[óo]rico)\b.{0,30}"
            r"(prontu[áa]rio|paciente|caso|interna[çc][ãa]o)"
            r"|\bme (conte|fale) sobre (o|a) paciente",
            re.IGNORECASE,
        ),
    ),
    (
        "consulta_paciente",
        re.compile(
            r"\b(quais|que|tem|possui|h[áa])\s+.{0,25}"
            r"(alergias?|comorbidades?|medica[çc][õo]es?|rem[ée]dios?)"
            r"|\bsinais vitais\b|\bqual (a )?(idade|creatinina|press[ãa]o)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass
class ResultadoTriagem:
    intencao: str
    metodo: str          # "heuristica" | "llm" | "padrao"
    justificativa: str = ""
    exige_paciente: bool = False
    sempre_validacao_humana: bool = False


def _instrucao_llm(intencoes: list[str]) -> str:
    return (
        "Classifique a pergunta do médico em EXATAMENTE UMA das intenções abaixo.\n\n"
        + "\n".join(f"- {i}" for i in intencoes)
        + "\n\nDefinições:\n"
        "- duvida_clinica: pergunta conceitual sobre doença, diagnóstico ou evidência, "
        "sem paciente específico envolvido.\n"
        "- consulta_paciente: pergunta sobre dados registrados de um paciente "
        "(alergias, medicações, comorbidades, sinais vitais).\n"
        "- exames_pendentes: pergunta sobre exames solicitados e ainda sem resultado.\n"
        "- conduta_terapeutica: pede sugestão de tratamento, dose, antibiótico ou manejo "
        "para um paciente.\n"
        "- resumo_prontuario: pede uma síntese do histórico clínico do paciente.\n\n"
        "REGRA DE DESEMPATE: na dúvida entre duas intenções, escolha a que exige mais "
        "cautela. Entre 'duvida_clinica' e 'conduta_terapeutica', escolha "
        "'conduta_terapeutica'. Classificar a menos faz o sistema pular uma etapa de "
        "validação de segurança; classificar a mais apenas acrescenta uma verificação.\n\n"
        "Responda com o identificador da intenção e mais nada."
    )


def classificar(pergunta: str, llm=None, *, tem_paciente: bool = False) -> ResultadoTriagem:
    """
    Classifica a intenção da pergunta.

    Args:
        llm: modelo a usar quando a heurística não decidir. Se None, cai na
            intenção padrão — o que mantém o fluxo funcionando sem LLM.
        tem_paciente: se há paciente vinculado à consulta. Uma pergunta sem
            paciente não pode ser `consulta_paciente`, por mais que o texto
            sugira.
    """
    pol = mod_politicas.carregar()
    permitidas = pol.intencoes_permitidas()

    def _montar(identificador: str, metodo: str, justificativa: str = "") -> ResultadoTriagem:
        definicao = pol.intencao(identificador) or {}
        resultado = ResultadoTriagem(
            intencao=identificador,
            metodo=metodo,
            justificativa=justificativa,
            exige_paciente=bool(definicao.get("exige_paciente")),
            sempre_validacao_humana=bool(definicao.get("sempre_validacao_humana")),
        )
        registrar(
            TipoEvento.DECISAO,
            f"Intenção classificada como '{identificador}' (via {metodo})",
            etapa="classificar_intencao",
            intencao=identificador,
            metodo=metodo,
            exige_paciente=resultado.exige_paciente,
            sempre_validacao_humana=resultado.sempre_validacao_humana,
        )
        return resultado

    # --- Nível 1: heurística ------------------------------------------------
    for identificador, padrao in HEURISTICAS:
        if not padrao.search(pergunta):
            continue
        # Intenções que exigem paciente não se aplicam a consultas soltas.
        definicao = pol.intencao(identificador) or {}
        if definicao.get("exige_paciente") and not tem_paciente:
            continue
        return _montar(identificador, "heuristica", f"padrão '{identificador}' reconhecido")

    # --- Nível 2: LLM -------------------------------------------------------
    if llm is None:
        return _montar(INTENCAO_PADRAO, "padrao", "sem LLM disponível para desempate")

    try:
        # Pipeline no formato do curso: prompt | llm, com .invoke().
        resposta = str(
            llm.invoke(
                [
                    SystemMessage(content=_instrucao_llm(permitidas)),
                    HumanMessage(content=pergunta),
                ]
            ).content
        ).strip().lower()
    except Exception as exc:
        log.warning("Falha ao classificar com LLM (%s) — usando a intenção padrão.", exc)
        return _montar(INTENCAO_PADRAO, "padrao", f"erro no LLM: {type(exc).__name__}")

    for identificador in permitidas:
        if identificador in resposta:
            definicao = pol.intencao(identificador) or {}
            if definicao.get("exige_paciente") and not tem_paciente:
                # O modelo escolheu uma intenção que exige paciente, mas não
                # há paciente. Rebaixamos para dúvida conceitual em vez de
                # falhar: a pergunta continua respondível pela evidência.
                return _montar(
                    INTENCAO_PADRAO, "llm",
                    f"LLM sugeriu '{identificador}', mas não há paciente vinculado",
                )
            return _montar(identificador, "llm", f"resposta do modelo: {resposta[:60]}")

    return _montar(INTENCAO_PADRAO, "padrao", f"resposta não reconhecida: {resposta[:60]}")
