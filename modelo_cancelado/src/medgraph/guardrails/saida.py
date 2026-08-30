"""
[REQ-3a][REQ-3c] Guardrail de saída.

O QUE FAZ:
    Verifica a resposta gerada ANTES de ela chegar ao médico, contra quatro
    invariantes:

      1. toda resposta clínica cita ao menos uma fonte;
      2. posologia só sai com marcação explícita de validação médica;
      3. o disclaimer institucional está presente;
      4. nenhum identificador de paciente vazou para o texto final.

POR QUE VERIFICAR A SAÍDA SE JÁ HÁ INSTRUÇÃO NO PROMPT DE SISTEMA:
    Porque instrução em prompt é pedido, não garantia. Um modelo de 3
    bilhões de parâmetros esquece o formato quando o contexto fica longo,
    quando a pergunta é atípica, ou simplesmente por variação de amostragem.

    A diferença entre pedir e verificar é a diferença entre um sistema que
    "normalmente" cita a fonte e um que não entrega resposta sem fonte. O
    requisito de explainability do enunciado é do segundo tipo.

O QUE ACONTECE COM UMA RESPOSTA REPROVADA:
    Ela não é descartada nem entregue. Volta para um nó de reescrita, com o
    motivo da reprovação anexado ao prompt, por até `max_tentativas`. Se ainda
    assim não passar, o sistema DEGRADA: entrega uma resposta mínima segura,
    que apresenta as fontes recuperadas sem afirmar nada sobre elas, e marca a
    consulta como degradada na trilha de auditoria.

    Degradar em vez de falhar é deliberado. O médico prefere receber "não
    consegui sintetizar, aqui estão os trechos relevantes" a receber um erro.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.dados.anonimizador import Anonimizador, Politica
from medgraph.guardrails import politicas as mod_politicas
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

# Frases que sinalizam que o modelo já marcou a necessidade de revisão. Basta
# uma delas para a exigência de posologia ser considerada atendida.
MARCAS_DE_VALIDACAO = (
    "validação do médico",
    "validacao do medico",
    "validação médica",
    "validacao medica",
    "médico responsável",
    "medico responsavel",
    "depende de validação",
    "depende de validacao",
    "pendente de validação",
    "aguardando validação",
    "não constitui prescrição",
    "nao constitui prescricao",
    "rascunho",
)


@dataclass
class Falha:
    """Uma invariante violada."""

    id: str
    descricao: str
    instrucao_correcao: str
    """Texto anexado ao prompt de reescrita. É o que torna o retry útil."""


@dataclass
class ResultadoSaida:
    """Veredito do guardrail de saída."""

    aprovado: bool
    resposta: str
    falhas: list[Falha] = field(default_factory=list)
    citacoes: list[str] = field(default_factory=list)
    tem_posologia: bool = False
    identificadores_vazados: dict[str, int] = field(default_factory=dict)

    @property
    def instrucoes_de_correcao(self) -> str:
        """Bloco a anexar ao prompt na tentativa de reescrita."""
        if not self.falhas:
            return ""
        itens = "\n".join(f"- {f.instrucao_correcao}" for f in self.falhas)
        return (
            "A resposta anterior foi reprovada na verificação de segurança. "
            "Corrija os pontos abaixo e reescreva:\n" + itens
        )

    def para_dict(self) -> dict[str, Any]:
        return {
            "aprovado": self.aprovado,
            "falhas": [f.id for f in self.falhas],
            "citacoes": self.citacoes,
            "tem_posologia": self.tem_posologia,
            "identificadores_vazados": self.identificadores_vazados,
        }


def verificar(
    resposta: str,
    *,
    marcadores_disponiveis: Sequence[str] = (),
    nomes_conhecidos: Sequence[str] = (),
    exigir_disclaimer: bool = False,
    cfg: Settings | None = None,
) -> ResultadoSaida:
    """
    Aplica as quatro invariantes à resposta gerada.

    Args:
        marcadores_disponiveis: marcadores que o recuperador realmente
            forneceu. Serve para detectar citação INVENTADA — o modelo citar
            [E7] quando só existiam [E1] e [P1] é uma alucinação de fonte, e
            é pior do que não citar: dá aparência de rastreabilidade a uma
            afirmação sem lastro.
        exigir_disclaimer: FALSO por padrão, e a razão importa.

            O disclaimer institucional é texto fixo. Exigir que o MODELO o
            escreva tem três problemas: gasta tokens de geração em conteúdo
            que já conhecemos; faz a aprovação depender de obediência a uma
            instrução; e, na prática, reprovava toda resposta do modelo base,
            que simplesmente não o incluía — jogando o fluxo no ciclo de
            reescrita a cada consulta.

            A garantia é mais forte quando o SISTEMA anexa o disclaimer, o
            que `grafo/nos.py::no_montar_resposta` faz de forma
            determinística. O guardrail passa a verificar apenas o que só o
            modelo pode cumprir: citar as fontes que recebeu, não inventar
            fontes, marcar a posologia e não vazar dado pessoal.

            Continua disponível como opção para quem quiser exigir o
            disclaimer na própria geração.
    """
    cfg = cfg or obter_settings()
    pol = mod_politicas.carregar()
    falhas: list[Falha] = []

    # --- 1. Citação de fonte  [REQ-3c] -------------------------------------
    citacoes: list[str] = []
    if pol.formato_citacao:
        citacoes = list(dict.fromkeys(pol.formato_citacao.findall(resposta)))
        # findall com grupo devolve só o grupo; recuperamos o marcador inteiro.
        citacoes = list(dict.fromkeys(re.findall(r"\[([EPC]\d+)\]", resposta)))

    minimo = int(pol.saida.get("minimo_citacoes", 1))
    if pol.saida.get("exigir_citacao", True) and len(citacoes) < minimo:
        falhas.append(
            Falha(
                id="sem_citacao",
                descricao=f"A resposta tem {len(citacoes)} citação(ões); o mínimo é {minimo}.",
                instrucao_correcao=(
                    "Cite a fonte de cada afirmação clínica usando os marcadores fornecidos "
                    f"no contexto ({', '.join(f'[{m}]' for m in marcadores_disponiveis) or '[E1]'}) "
                    "e encerre com uma linha 'Fontes:' listando os marcadores usados."
                ),
            )
        )

    # Citação de marcador inexistente — alucinação de fonte.
    if marcadores_disponiveis:
        disponiveis = set(marcadores_disponiveis)
        inventadas = [c for c in citacoes if c not in disponiveis]
        if inventadas:
            falhas.append(
                Falha(
                    id="citacao_inexistente",
                    descricao=f"Citou fonte(s) que não foram fornecidas: {inventadas}.",
                    instrucao_correcao=(
                        "Use APENAS os marcadores presentes no contexto fornecido: "
                        f"{', '.join(f'[{m}]' for m in sorted(disponiveis))}. "
                        f"Os marcadores {inventadas} não existem."
                    ),
                )
            )

    # --- 2. Posologia sem marcação de revisão  [REQ-3a] --------------------
    tem_posologia = any(p.search(resposta) for p in pol.padroes_posologia)
    if tem_posologia and pol.saida.get("proibir_posologia_sem_revisao", True):
        minuscula = resposta.lower()
        if not any(marca in minuscula for marca in MARCAS_DE_VALIDACAO):
            falhas.append(
                Falha(
                    id="posologia_sem_revisao",
                    descricao="A resposta menciona dose ou posologia sem marcar a validação médica.",
                    instrucao_correcao=(
                        "A resposta menciona dose, via ou posologia. Acrescente explicitamente "
                        "que se trata de informação de protocolo e que a prescrição depende de "
                        "validação do médico responsável."
                    ),
                )
            )

    # --- 3. Disclaimer institucional ---------------------------------------
    if exigir_disclaimer and pol.saida.get("exigir_disclaimer", True):
        # Comparamos por palavras-chave, e não pelo texto inteiro: exigir a
        # frase literal faria o modelo gastar tokens copiando o parágrafo, e
        # qualquer variação mínima reprovaria a resposta.
        marcas_disclaimer = ("não substitui", "nao substitui", "sugestão", "sugestao", "apoio à decisão")
        if not any(m in resposta.lower() for m in marcas_disclaimer):
            falhas.append(
                Falha(
                    id="sem_disclaimer",
                    descricao="Falta a ressalva de que a resposta não substitui o julgamento clínico.",
                    instrucao_correcao=(
                        "Encerre deixando claro que é uma sugestão de apoio à decisão e que "
                        "não substitui o julgamento do médico assistente."
                    ),
                )
            )

    # --- 4. Vazamento de dado pessoal  [REQ-1a] ----------------------------
    identificadores: dict[str, int] = {}
    if pol.saida.get("proibir_pii_na_resposta", True):
        anonimizador = Anonimizador(
            politica=Politica.MASCARAR, nomes_conhecidos=list(nomes_conhecidos)
        )
        anonimizador.redigir(resposta)
        if anonimizador.total_removido:
            identificadores = anonimizador.estatisticas()
            falhas.append(
                Falha(
                    id="pii_na_resposta",
                    descricao=f"A resposta contém identificador(es): {identificadores}.",
                    instrucao_correcao=(
                        "Remova qualquer dado pessoal identificável. Refira-se ao paciente "
                        "apenas pelo identificador interno do prontuário."
                    ),
                )
            )

    resultado = ResultadoSaida(
        aprovado=not falhas,
        resposta=resposta,
        falhas=falhas,
        citacoes=citacoes,
        tem_posologia=tem_posologia,
        identificadores_vazados=identificadores,
    )

    registrar(
        TipoEvento.GUARDRAIL,
        "Saída aprovada" if resultado.aprovado else f"Saída reprovada: {[f.id for f in falhas]}",
        nivel="INFO" if resultado.aprovado else "WARNING",
        etapa="guardrail_saida",
        **resultado.para_dict(),
    )
    return resultado


def resposta_degradada(
    marcadores: Sequence[str],
    trechos: Sequence[dict[str, str]] = (),
    cfg: Settings | None = None,
) -> str:
    """
    Resposta mínima segura, usada quando a reescrita esgota as tentativas.

    Apresenta as fontes recuperadas SEM afirmar nada sobre elas. O médico
    continua com material útil — os trechos relevantes do protocolo e da
    literatura — e o sistema não entrega uma síntese que não conseguiu
    validar.

    É melhor do que devolver um erro: o médico está com uma dúvida clínica
    em aberto, e "não consegui sintetizar, mas aqui está o que encontrei"
    resolve mais do que "ocorreu uma falha".
    """
    cfg = cfg or obter_settings()
    pol = mod_politicas.carregar()

    linhas = [
        "Não foi possível produzir uma síntese que atendesse aos critérios de segurança "
        "do assistente. Para não entregar uma resposta não verificada, apresento apenas "
        "as fontes recuperadas para a sua consulta:",
        "",
    ]
    if trechos:
        for trecho in trechos:
            linhas.append(f"[{trecho['marcador']}] {trecho['titulo']}")
            linhas.append(f"    {trecho['texto'][:400].strip()}...")
            linhas.append("")
    elif marcadores:
        linhas.append("Fontes recuperadas: " + ", ".join(f"[{m}]" for m in marcadores))
        linhas.append("")

    linhas.append(f"Fontes: {', '.join(f'[{m}]' for m in marcadores) or '(nenhuma)'}")
    linhas.append("")
    linhas.append(pol.texto("disclaimer"))

    registrar(
        TipoEvento.GUARDRAIL,
        "Resposta degradada emitida — tentativas de reescrita esgotadas",
        nivel="WARNING",
        etapa="guardrail_saida",
        marcadores=list(marcadores),
    )
    return "\n".join(linhas)


def acrescentar_disclaimer(resposta: str, cfg: Settings | None = None) -> str:
    """Anexa o disclaimer institucional, se ainda não estiver presente."""
    pol = mod_politicas.carregar()
    disclaimer = pol.texto("disclaimer")
    if "não substitui" in resposta.lower() or "nao substitui" in resposta.lower():
        return resposta
    return f"{resposta.rstrip()}\n\n---\n{disclaimer}"
