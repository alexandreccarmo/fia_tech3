"""
Fluxo de decisao clinica em LangGraph.

O enunciado pede "fluxos de decisao automatizados e seguros, onde, ao receber
informacoes sobre um paciente, o sistema possa acionar diferentes etapas, como
verificar exames pendentes, sugerir tratamentos e emitir alertas".

Sao seis nos. Cada um recebe o estado, devolve o estado modificado, e registra
o que fez - o log detalhado que o item 3 exige sai desse registro, e nao de
prints espalhados.

    guardrail_entrada
        |
        +--(recusado)--> montar_resposta
        |
    consultar_prontuario -> recuperar_evidencia -> responder
        |
    verificar_resposta
        |
        +--(risco alto)--> validacao_humana --> montar_resposta
        |
        +--(ok)---------------------------> montar_resposta
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import auditoria, guardrails, prontuario, rag


class Estado(TypedDict, total=False):
    """O que atravessa o grafo. Tudo o que um no produz fica visivel aos demais."""

    pergunta: str
    paciente_id: str
    paciente: Any
    trechos: list
    resposta: str
    achados: list
    aguardando_validacao: bool
    recusado: bool
    trilha: list[dict]
    trace_id: str


def _registrar(estado: Estado, etapa: str, detalhe: str, inicio: float,
               nivel: str = "INFO") -> None:
    """
    Registra o evento nos dois lugares: no estado e na trilha persistida.

    No estado porque os graficos leem dali; em arquivo porque o item 3 do
    enunciado pede rastreamento e auditoria, e um log que existe so na memoria
    do processo nao audita nada depois que ele termina.
    """
    ms = (time.perf_counter() - inicio) * 1000
    estado.setdefault("trilha", []).append({
        "etapa": etapa,
        "detalhe": detalhe,
        "ms": round(ms, 1),
        "nivel": nivel,
    })
    trilha = auditoria.trilha_atual()
    if trilha is not None:
        trilha.registrar(etapa, detalhe, ms, nivel)


def construir(indice, responder_fn, caminho_banco: str):
    """
    Monta o grafo. `responder_fn` recebe (pergunta, contexto) e devolve texto.

    A funcao de resposta entra por parametro para que o grafo nao saiba qual
    modelo responde - o mesmo fluxo roda com o modelo base ou com o ajustado, o
    que e justamente o que a comparacao da avaliacao precisa.
    """

    def no_guardrail_entrada(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        verificacao = guardrails.verificar_entrada(estado["pergunta"])
        estado["recusado"] = not verificacao.aprovado
        if estado["recusado"]:
            estado["resposta"] = (
                "Nao posso atender a esse pedido. O assistente apresenta evidencia "
                "e devolve a decisao ao medico responsavel."
            )
            estado["achados"] = verificacao.achados
        _registrar(estado, "guardrail_entrada",
                   "recusado" if estado["recusado"] else "aprovado", inicio,
                   "ALERTA" if estado["recusado"] else "INFO")
        return estado

    def no_consultar_prontuario(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        paciente = None
        if estado.get("paciente_id"):
            paciente = prontuario.buscar(estado["paciente_id"], caminho_banco)
        estado["paciente"] = paciente
        _registrar(estado, "consultar_prontuario",
                   paciente.id if paciente else "sem paciente vinculado", inicio)
        return estado

    def no_recuperar_evidencia(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        estado["trechos"] = rag.recuperar(indice, estado["pergunta"], k=2)
        marcadores = ", ".join(t.marcador for t in estado["trechos"])
        _registrar(estado, "recuperar_evidencia", f"fontes: {marcadores}", inicio)
        return estado

    def no_responder(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        contexto = rag.montar_contexto(estado["trechos"])
        if estado.get("paciente"):
            contexto = f"{estado['paciente'].resumo()}\n\n{contexto}"
        estado["resposta"] = responder_fn(estado["pergunta"], contexto)
        _registrar(estado, "responder", f"{len(estado['resposta'])} caracteres", inicio)
        return estado

    def no_verificar(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        verificacao = guardrails.verificar_resposta(
            estado["resposta"], estado.get("paciente")
        )
        estado["achados"] = verificacao.achados
        estado["aguardando_validacao"] = bool(verificacao.criticos)
        nivel = "CRITICO" if verificacao.criticos else (
            "ALERTA" if verificacao.achados else "INFO"
        )
        _registrar(estado, "verificar_resposta",
                   f"{len(verificacao.achados)} achado(s), "
                   f"{len(verificacao.criticos)} critico(s)", inicio, nivel)
        return estado

    def no_validacao_humana(estado: Estado) -> Estado:
        """
        Aqui a execucao PARA.

        E o ponto do enunciado que diz "nunca prescrever diretamente, sem
        validacao humana": quando ha conflito critico, a resposta e retida e o
        medico decide.
        """
        inicio = time.perf_counter()
        _registrar(estado, "validacao_humana", "resposta retida para revisao",
                   inicio, "CRITICO")
        return estado

    def no_montar_resposta(estado: Estado) -> Estado:
        inicio = time.perf_counter()
        partes = [estado["resposta"]]
        if estado.get("trechos") and not estado.get("recusado"):
            fontes = "; ".join(f"[{t.marcador}] {t.titulo}" for t in estado["trechos"])
            partes.append(f"\nFontes: {fontes}")
        if estado.get("aguardando_validacao"):
            partes.append("\n[RETIDA] Aguardando validacao de medico responsavel.")
        partes.append(guardrails.DISCLAIMER)
        estado["resposta"] = "".join(partes)
        _registrar(estado, "montar_resposta", "resposta final montada", inicio)
        return estado

    grafo = StateGraph(Estado)
    grafo.add_node("guardrail_entrada", no_guardrail_entrada)
    grafo.add_node("consultar_prontuario", no_consultar_prontuario)
    grafo.add_node("recuperar_evidencia", no_recuperar_evidencia)
    grafo.add_node("responder", no_responder)
    grafo.add_node("verificar_resposta", no_verificar)
    grafo.add_node("validacao_humana", no_validacao_humana)
    grafo.add_node("montar_resposta", no_montar_resposta)

    grafo.set_entry_point("guardrail_entrada")
    grafo.add_conditional_edges(
        "guardrail_entrada",
        lambda e: "montar_resposta" if e.get("recusado") else "consultar_prontuario",
        {"montar_resposta": "montar_resposta", "consultar_prontuario": "consultar_prontuario"},
    )
    grafo.add_edge("consultar_prontuario", "recuperar_evidencia")
    grafo.add_edge("recuperar_evidencia", "responder")
    grafo.add_edge("responder", "verificar_resposta")
    grafo.add_conditional_edges(
        "verificar_resposta",
        lambda e: "validacao_humana" if e.get("aguardando_validacao") else "montar_resposta",
        {"validacao_humana": "validacao_humana", "montar_resposta": "montar_resposta"},
    )
    grafo.add_edge("validacao_humana", "montar_resposta")
    grafo.add_edge("montar_resposta", END)

    return grafo.compile()


def consultar(app, pergunta: str, paciente_id: str | None = None,
              arquivo_auditoria: str | None = None, console: bool = False) -> Estado:
    """
    Executa uma consulta, com trilha de auditoria propria.

    Cada consulta ganha um trace_id: sem ele os eventos de consultas diferentes
    se misturariam no arquivo, e a trilha deixaria de reconstruir o que
    aconteceu em cada uma.
    """
    trilha = auditoria.TrilhaAuditoria(
        arquivo=arquivo_auditoria or auditoria.ARQUIVO_PADRAO, console=console
    )
    auditoria.definir_trilha(trilha)
    try:
        estado = app.invoke({"pergunta": pergunta, "paciente_id": paciente_id or ""})
    finally:
        auditoria.definir_trilha(None)
    estado["trace_id"] = trilha.trace_id
    return estado
