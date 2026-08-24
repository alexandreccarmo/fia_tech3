"""
[REQ-E1] Execução do fluxo clínico.

O QUE FAZ:
    Envolve a invocação do grafo com o que ela precisa ao redor: abertura da
    trilha de auditoria, identificação da thread para o checkpointer, e o
    tratamento da pausa para validação humana.

POR QUE UMA CAMADA ACIMA DO GRAFO:
    Quem chama o assistente — a linha de comando, o painel Streamlit, um
    teste — não deveria precisar saber que existe um `thread_id`, nem como
    detectar que a execução parou numa interrupção, nem como retomá-la. Aqui
    isso vira duas funções: `consultar()` e `validar()`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import Desfecho, TipoEvento, abrir_trilha, registrar
from medgraph.grafo.construir import NOME_ARQUIVO_CHECKPOINT, obter_grafo
from medgraph.grafo.estado import EstadoClinico, estado_inicial
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


@dataclass
class Consulta:
    """O resultado de uma consulta ao assistente."""

    trace_id: str
    thread_id: str
    estado: dict[str, Any]
    pausada: bool = False
    """True quando o fluxo parou aguardando validação médica."""

    etapas: list[str] = field(default_factory=list)
    tempo_por_etapa: dict[str, float] = field(default_factory=dict)
    duracao_ms: float = 0.0

    @property
    def resposta(self) -> str:
        if self.pausada:
            return (
                "Resposta retida: esta consulta foi classificada como de alto risco e "
                "aguarda validação de um médico responsável antes de ser liberada."
            )
        return self.estado.get("resposta_final", "")

    @property
    def desfecho(self) -> str:
        if self.pausada:
            return "aguardando_validacao"
        return self.estado.get("desfecho", "respondida")

    @property
    def alertas(self) -> list[dict[str, Any]]:
        return self.estado.get("alertas", [])

    @property
    def fontes(self) -> list[dict[str, Any]]:
        return self.estado.get("fontes_citadas", []) or self.estado.get("trechos", [])


def _config_thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def consultar(
    pergunta: str,
    *,
    paciente_id: str | None = None,
    usuario: str = "medico",
    thread_id: str | None = None,
    cfg: Settings | None = None,
) -> Consulta:
    """
    Executa uma consulta clínica completa.

    Se o fluxo parar para validação humana, `Consulta.pausada` volta True e o
    `thread_id` permite retomar depois com `validar()`.
    """
    cfg = cfg or obter_settings()
    thread_id = thread_id or uuid.uuid4().hex[:16]
    app = obter_grafo(cfg)

    with abrir_trilha(
        pergunta=pergunta, usuario=usuario, paciente_id=paciente_id, cfg=cfg
    ) as trilha:
        inicial: EstadoClinico = estado_inicial(
            pergunta, paciente_id=paciente_id, usuario=usuario
        )
        app.invoke(inicial, config=_config_thread(thread_id))

        # Depois de invocar, o estado autoritativo é o do checkpointer: ele
        # reflete a execução mesmo quando ela parou numa interrupção, caso em
        # que o retorno do invoke() é apenas parcial.
        instantaneo = app.get_state(_config_thread(thread_id))
        estado = dict(instantaneo.values)
        pausada = bool(instantaneo.next)  # há próximo nó pendente = parou

        if pausada:
            trilha.desfecho = Desfecho.AGUARDANDO_VALIDACAO
        elif estado.get("desfecho") == "recusada":
            trilha.desfecho = Desfecho.RECUSADA
        elif estado.get("desfecho") == "degradada":
            trilha.desfecho = Desfecho.DEGRADADA

        consulta = Consulta(
            trace_id=trilha.trace_id,
            thread_id=thread_id,
            estado=estado,
            pausada=pausada,
            etapas=trilha.etapas_executadas(),
            tempo_por_etapa=trilha.tempo_por_etapa(),
        )

    consulta.duracao_ms = sum(consulta.tempo_por_etapa.values())
    return consulta


def validar(
    thread_id: str,
    *,
    validado_por: str,
    parecer: str = "",
    cfg: Settings | None = None,
) -> Consulta:
    """
    Registra a validação médica e retoma o fluxo interrompido.  [REQ-3a]

    COMO A RETOMADA FUNCIONA:
        `update_state` grava quem validou no estado persistido; `invoke(None)`
        continua a execução do ponto em que parou. O `None` é o que diz ao
        LangGraph "não comece uma execução nova, retome a existente".

    POR QUE O EVENTO DE AUDITORIA É REGISTRADO AQUI, E NÃO SÓ NO NÓ:
        Descobrimos, testando a retomada, que o nó `aguardar_validacao` NÃO
        executa. O `update_state()` grava os valores como se viessem do nó
        pendente, o que faz o LangGraph considerar aquela tarefa cumprida e
        pular direto para a seguinte. O corpo do nó — e portanto o evento
        VALIDACAO_HUMANA que ele registrava — nunca rodava.

        O registro de QUEM validou uma conduta clínica de alto risco é o
        artefato de auditoria mais importante deste fluxo. Ele não pode
        depender de um detalhe do mecanismo de retomada de uma biblioteca.
        Passa a ser registrado aqui, explicitamente, antes da retomada.

    Esta função é a contrapartida do requisito "nunca prescrever sem validação
    humana": a resposta só é liberada depois que uma pessoa identificada
    passou por aqui.
    """
    cfg = cfg or obter_settings()
    app = obter_grafo(cfg)
    config = _config_thread(thread_id)

    with abrir_trilha(
        pergunta=f"[validação da thread {thread_id}]", usuario=validado_por, cfg=cfg
    ) as trilha:
        anterior = app.get_state(config)
        registrar(
            TipoEvento.VALIDACAO_HUMANA,
            f"Validação registrada por {validado_por}",
            etapa="aguardar_validacao",
            conclusao=True,
            thread_id=thread_id,
            validado_por=validado_por,
            parecer=parecer,
            escore_risco=anterior.values.get("escore_risco"),
            gatilhos=anterior.values.get("gatilhos_risco", []),
            achados=[a["titulo"] for a in anterior.values.get("achados_clinicos", [])],
            pergunta=anterior.values.get("pergunta", ""),
            paciente_id=anterior.values.get("paciente_id"),
        )

        app.update_state(
            config,
            {"validado_por": validado_por, "parecer_validacao": parecer},
        )
        app.invoke(None, config=config)

        instantaneo = app.get_state(config)
        consulta = Consulta(
            trace_id=trilha.trace_id,
            thread_id=thread_id,
            estado=dict(instantaneo.values),
            pausada=bool(instantaneo.next),
            etapas=trilha.etapas_executadas(),
            tempo_por_etapa=trilha.tempo_por_etapa(),
        )

    consulta.duracao_ms = sum(consulta.tempo_por_etapa.values())
    log.info("Consulta %s validada por %s", thread_id, validado_por)
    return consulta


def consultas_pendentes(cfg: Settings | None = None) -> list[dict[str, Any]]:
    """
    Lista as consultas paradas aguardando validação.

    Alimenta a fila de validação do painel. É o que impede que uma consulta
    retida seja simplesmente esquecida — quem perguntou vê a resposta
    pendente, e o responsável vê a fila.

    NOTA DE IMPLEMENTAÇÃO:
        A primeira versão percorria `checkpointer.list()` diretamente e
        quebrava: aquele método devolve tuplas de checkpoint bruto, não os
        `StateSnapshot` que expõem o campo `next`. O caminho correto é
        `app.get_state()` por thread — a API pública, que monta o snapshot
        completo.

        Como o checkpointer não oferece "liste todas as threads", as threads
        conhecidas são lidas da própria tabela do SQLite. É acoplamento ao
        formato interno, e está isolado aqui justamente por isso.
    """
    cfg = cfg or obter_settings()
    app = obter_grafo(cfg)

    caminho = cfg.dir_logs / NOME_ARQUIVO_CHECKPOINT
    if not caminho.exists():
        return []

    import sqlite3

    try:
        conexao = sqlite3.connect(f"file:{caminho}?mode=ro", uri=True)
        try:
            threads = [
                linha[0]
                for linha in conexao.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC"
                )
            ]
        finally:
            conexao.close()
    except sqlite3.Error as exc:
        log.warning("Não foi possível ler as threads do checkpointer: %s", exc)
        return []

    pendentes: list[dict[str, Any]] = []
    for thread_id in threads:
        try:
            instantaneo = app.get_state(_config_thread(thread_id))
        except Exception:
            continue
        if not instantaneo.next:
            continue

        estado = instantaneo.values
        pendentes.append({
            "thread_id": thread_id,
            "pergunta": estado.get("pergunta", ""),
            "paciente_id": estado.get("paciente_id"),
            "escore_risco": estado.get("escore_risco"),
            "gatilhos": estado.get("gatilhos_risco", []),
            "alertas": estado.get("alertas", []),
            "proximo_no": list(instantaneo.next),
        })

    return pendentes
