"""
[REQ-E1] Estado compartilhado do fluxo clínico.

O QUE É:
    O contrato de dados que circula entre os nós do LangGraph. Tudo o que um
    nó produz e outro consome passa por aqui — e nada mais.

POR QUE UM TypedDict, COMO NAS AULAS:
    É o padrão que o curso apresentou, e é o certo para este caso. Um nó do
    LangGraph recebe o estado e devolve um DELTA — um dicionário parcial com
    apenas as chaves que ele alterou. O LangGraph funde esse delta no estado.
    `total=False` é o que torna isso possível: nenhuma chave é obrigatória.

O CAMPO `historico` E O AGREGADOR:
    Vários nós podem escrever no histórico no mesmo passo do grafo. Sem um
    agregador, o LangGraph levanta `InvalidUpdateError` — duas escritas
    concorrentes na mesma chave. `Annotated[list, add_messages]` resolve
    dizendo ao grafo como COMBINAR as escritas em vez de escolher uma.

    Foi exatamente o problema apresentado no Vídeo 1 da Aula 1 de LangGraph.

O QUE DELIBERADAMENTE NÃO ESTÁ NO ESTADO:
    O objeto `Paciente` completo. O estado é serializado no checkpointer e
    aparece na trilha de auditoria; carregar o registro clínico inteiro em
    todos os passos significaria gravá-lo repetidas vezes em disco. Guardamos
    o RESUMO já anonimizado (`contexto_paciente`), que é o que o modelo lê, e
    um dicionário reduzido para o painel.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Desfecho = Literal[
    "respondida",
    "recusada",
    "aguardando_validacao",
    "degradada",
    "erro",
]


class EstadoClinico(TypedDict, total=False):
    """Estado que circula pelos quatorze nós do fluxo."""

    # -------------------------------------------------------------------------
    # ENTRADA
    # -------------------------------------------------------------------------
    pergunta: str
    """A pergunta como o médico escreveu, sem tratamento."""

    paciente_id: str | None
    """Identificador do paciente, quando a consulta é vinculada a um."""

    usuario: str
    """Quem está consultando. Vai para a trilha de auditoria."""

    # -------------------------------------------------------------------------
    # NÓ 1 — guardrail de entrada
    # -------------------------------------------------------------------------
    pergunta_limpa: str
    """A pergunta após a remoção de identificadores. É esta que vai ao modelo."""

    aprovado_entrada: bool
    motivo_recusa: str
    id_bloqueio: str
    emergencia: bool
    termos_emergencia: list[str]

    # -------------------------------------------------------------------------
    # NÓ 2 — classificação de intenção
    # -------------------------------------------------------------------------
    intencao: str
    metodo_intencao: str
    exige_paciente: bool
    intencao_exige_validacao: bool
    """True quando a política marca a intenção como sempre sujeita a validação."""

    # -------------------------------------------------------------------------
    # NÓ 3 — prontuário
    # -------------------------------------------------------------------------
    contexto_paciente: str
    """Resumo clínico anonimizado, injetado no prompt como fonte [C1]."""

    paciente_resumo: dict[str, Any]
    """Versão estruturada e reduzida, para o painel visual."""

    exames_pendentes: list[str]
    paciente_encontrado: bool

    # -------------------------------------------------------------------------
    # NÓ 4 — recuperação de evidência
    # -------------------------------------------------------------------------
    trechos: list[dict[str, Any]]
    """Trechos recuperados, serializados. Cada um traz seu marcador de citação."""

    marcadores: list[str]
    """Marcadores disponíveis: ['E1', 'P1', 'C1']. O guardrail de saída os usa
    para detectar citação de fonte inexistente."""

    evidencia_suficiente: bool

    # -------------------------------------------------------------------------
    # NÓ 5 — raciocínio clínico
    # -------------------------------------------------------------------------
    resposta_bruta: str
    tentativas_reescrita: int
    provedor_llm: str

    # -------------------------------------------------------------------------
    # NÓ 6 — regras clínicas
    # -------------------------------------------------------------------------
    achados_clinicos: list[dict[str, Any]]
    risco_clinico: float
    tem_bloqueio_clinico: bool
    farmacos_detectados: list[str]

    # -------------------------------------------------------------------------
    # NÓ 7 — guardrail de saída
    # -------------------------------------------------------------------------
    aprovado_saida: bool
    falhas_saida: list[str]
    instrucoes_correcao: str
    citacoes_usadas: list[str]

    # -------------------------------------------------------------------------
    # NÓ 9 — triagem de risco e validação humana
    # -------------------------------------------------------------------------
    escore_risco: float
    """Risco agregado, de 0 a 1. Combina os achados clínicos e os gatilhos
    de contexto definidos em politicas.yaml."""

    gatilhos_risco: list[str]
    exige_validacao_humana: bool
    validado_por: str
    parecer_validacao: str

    # -------------------------------------------------------------------------
    # NÓ 10 — alertas
    # -------------------------------------------------------------------------
    alertas: list[dict[str, Any]]

    # -------------------------------------------------------------------------
    # NÓ 11 — saída
    # -------------------------------------------------------------------------
    resposta_final: str
    desfecho: Desfecho
    fontes_citadas: list[dict[str, Any]]

    # -------------------------------------------------------------------------
    # Transversal
    # -------------------------------------------------------------------------
    historico: Annotated[list, add_messages]
    """Registro das passagens pelos nós. Usa agregador porque mais de um nó
    pode escrever aqui no mesmo passo."""

    erro: str


def estado_inicial(
    pergunta: str,
    *,
    paciente_id: str | None = None,
    usuario: str = "anonimo",
) -> EstadoClinico:
    """
    Monta o estado de partida de uma consulta.

    Os contadores e as listas são inicializados explicitamente. Deixar que
    surjam sozinhos no primeiro uso faria cada nó ter que testar a existência
    da chave antes de somar ou anexar — e o primeiro que esquecesse quebraria
    o fluxo com um KeyError no meio da execução.
    """
    return EstadoClinico(
        pergunta=pergunta,
        paciente_id=paciente_id,
        usuario=usuario,
        tentativas_reescrita=0,
        trechos=[],
        marcadores=[],
        achados_clinicos=[],
        alertas=[],
        gatilhos_risco=[],
        falhas_saida=[],
        citacoes_usadas=[],
        historico=[],
    )
