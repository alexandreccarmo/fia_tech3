"""
[REQ-E1] Montagem do grafo LangGraph.

O QUE FAZ:
    Liga os quatorze nós com as arestas lineares e condicionais, configura o
    checkpointer e compila o fluxo executável.

O DESENHO DO FLUXO:

    guardrail_entrada
        ├── (bloqueado) → responder_recusa → END
        └── (aprovado)  → classificar_intencao
                              ├── (com paciente) → consultar_prontuario ─┐
                              └── (sem paciente) ──────────────────────► recuperar_evidencia
                                                                              │
                                                          raciocinio_clinico ◄┘
                                                                  │
                                                          regras_clinicas
                                                                  │
                                                          guardrail_saida
                                            ┌──────────────────┼──────────────────┐
                                    (reprovado,          (reprovado,         (aprovado)
                                    há tentativa)      esgotado)                  │
                                          │                   │                   │
                                     reescrever      degradar_resposta            │
                                          │                   │                   │
                                          └──► raciocinio_clinico                 │
                                                              └───────────────────┤
                                                                                  ▼
                                                                          triagem_risco
                                                                                     │
                                                                            emitir_alertas
                                                          ┌───────────────────────┴────┐
                                                    (alto risco)              (baixo risco)
                                                          │                          │
                                                 aguardar_validacao ────────────────►│
                                                                                     ▼
                                                                            montar_resposta
                                                                                     │
                                                                                    END

POR QUE UM CHECKPOINTER SQLITE:
    A validação humana exige que o estado sobreviva ao fim do processo. O
    médico pode validar minutos depois, de outra aba do painel, ou no dia
    seguinte. Um checkpointer em memória perderia a consulta pendente ao
    fechar o Streamlit.

    Cada consulta é uma `thread` identificada pelo `thread_id`. Retomar é
    invocar o grafo de novo com o mesmo identificador.

POR QUE `interrupt_before` E NÃO UM NÓ QUE APENAS MARCA:
    Marcar "aguardando validação" no estado e seguir em frente seria teatro:
    a resposta chegaria ao médico com um aviso, mas teria chegado. Com
    `interrupt_before`, a execução PARA antes do nó. O requisito "nunca
    prescrever sem validação humana" vira uma propriedade da execução, não
    uma frase no texto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from config.settings import Settings, obter_settings
from medgraph.grafo import nos, rotas
from medgraph.grafo.estado import EstadoClinico
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

NOME_ARQUIVO_CHECKPOINT = "checkpoints_grafo.sqlite"


def montar_grafo() -> StateGraph:
    """
    Declara nós e arestas. Não compila — permite inspecionar antes.

    Separar a declaração da compilação é o que torna possível gerar os
    diagramas sem instanciar um checkpointer, e testar a topologia do fluxo
    sem executá-lo.
    """
    grafo = StateGraph(EstadoClinico)

    # --- nós ---------------------------------------------------------------
    grafo.add_node("guardrail_entrada", nos.no_guardrail_entrada)
    grafo.add_node("responder_recusa", nos.no_responder_recusa)
    grafo.add_node("classificar_intencao", nos.no_classificar_intencao)
    grafo.add_node("consultar_prontuario", nos.no_consultar_prontuario)
    grafo.add_node("recuperar_evidencia", nos.no_recuperar_evidencia)
    grafo.add_node("raciocinio_clinico", nos.no_raciocinio_clinico)
    grafo.add_node("regras_clinicas", nos.no_regras_clinicas)
    grafo.add_node("guardrail_saida", nos.no_guardrail_saida)
    grafo.add_node("reescrever", nos.no_reescrever)
    grafo.add_node("degradar_resposta", nos.no_degradar_resposta)
    grafo.add_node("triagem_risco", nos.no_triagem_risco)
    grafo.add_node("aguardar_validacao", nos.no_aguardar_validacao)
    grafo.add_node("emitir_alertas", nos.no_emitir_alertas)
    grafo.add_node("montar_resposta", nos.no_montar_resposta)

    grafo.set_entry_point("guardrail_entrada")

    # --- bifurcação 1: entrada aprovada? -----------------------------------
    grafo.add_conditional_edges(
        "guardrail_entrada",
        rotas.apos_guardrail_entrada,
        {
            "responder_recusa": "responder_recusa",
            "classificar_intencao": "classificar_intencao",
        },
    )
    grafo.add_edge("responder_recusa", END)

    # --- bifurcação 2: precisa do prontuário? ------------------------------
    grafo.add_conditional_edges(
        "classificar_intencao",
        rotas.apos_classificar_intencao,
        {
            "consultar_prontuario": "consultar_prontuario",
            "recuperar_evidencia": "recuperar_evidencia",
        },
    )
    grafo.add_edge("consultar_prontuario", "recuperar_evidencia")

    # --- trecho linear ------------------------------------------------------
    grafo.add_edge("recuperar_evidencia", "raciocinio_clinico")
    grafo.add_edge("raciocinio_clinico", "regras_clinicas")
    grafo.add_edge("regras_clinicas", "guardrail_saida")

    # --- bifurcação 3: o ciclo de reescrita --------------------------------
    grafo.add_conditional_edges(
        "guardrail_saida",
        rotas.apos_guardrail_saida,
        {
            "triagem_risco": "triagem_risco",
            "reescrever": "reescrever",
            "degradar_resposta": "degradar_resposta",
        },
    )
    # A aresta que fecha o ciclo. É a única do grafo que volta.
    grafo.add_edge("reescrever", "raciocinio_clinico")
    grafo.add_edge("degradar_resposta", "triagem_risco")

    # --- alertas SEMPRE, e ANTES da validação -------------------------------
    # A ordem foi corrigida durante os testes de integração. Na primeira
    # versão os alertas eram emitidos depois da validação humana, o que
    # deixava o médico validador sem a informação de que mais precisa: quais
    # conflitos de segurança foram detectados. Validar às cegas seria pior do
    # que não validar, porque produziria um registro de aprovação sem
    # fundamento.
    grafo.add_edge("triagem_risco", "emitir_alertas")

    # --- bifurcação 4: validação humana ------------------------------------
    grafo.add_conditional_edges(
        "emitir_alertas",
        rotas.apos_triagem_risco,
        {
            "aguardar_validacao": "aguardar_validacao",
            "montar_resposta": "montar_resposta",
        },
    )
    grafo.add_edge("aguardar_validacao", "montar_resposta")

    # --- fechamento ---------------------------------------------------------
    grafo.add_edge("montar_resposta", END)

    return grafo


def compilar(
    cfg: Settings | None = None,
    *,
    com_checkpointer: bool = True,
    com_validacao_humana: bool = True,
):
    """
    Compila o grafo pronto para executar.

    Args:
        com_checkpointer: persiste o estado em SQLite. Necessário para a
            validação humana; desligado nos testes que não a exercitam.
        com_validacao_humana: ativa a pausa antes do nó de validação. Quando
            False, o fluxo atravessa sem parar — usado nos testes de
            integração que verificam o caminho completo.
    """
    cfg = cfg or obter_settings()
    grafo = montar_grafo()

    opcoes: dict[str, Any] = {}

    if com_checkpointer:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        caminho: Path = cfg.dir_logs / NOME_ARQUIVO_CHECKPOINT
        caminho.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False porque o Streamlit atende requisições em
        # threads diferentes; sem isso o SQLite recusa a conexão reutilizada.
        conexao = sqlite3.connect(str(caminho), check_same_thread=False)
        opcoes["checkpointer"] = SqliteSaver(conexao)

    if com_validacao_humana and com_checkpointer:
        # A interrupção só funciona com checkpointer: sem ele não há onde
        # guardar o estado da execução pausada.
        opcoes["interrupt_before"] = ["aguardar_validacao"]

    compilado = grafo.compile(**opcoes)
    log.info(
        "Grafo compilado: %d nós | checkpointer=%s | validação humana=%s",
        len(grafo.nodes),
        com_checkpointer,
        com_validacao_humana and com_checkpointer,
    )
    return compilado


# Instância única por processo, criada sob demanda.
_grafo_compilado = None


def obter_grafo(cfg: Settings | None = None):
    """Devolve o grafo compilado, reaproveitando a instância do processo."""
    global _grafo_compilado
    if _grafo_compilado is None:
        _grafo_compilado = compilar(cfg)
    return _grafo_compilado
