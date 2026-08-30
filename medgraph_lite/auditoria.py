"""
Logging e trilha de auditoria.

O item 3 do enunciado pede "logging detalhado para rastreamento e auditoria".
Sao dois publicos diferentes, e por isso dois destinos:

    CONSOLE   para quem esta assistindo agora - colorido, uma linha por etapa
    ARQUIVO   para quem vai auditar depois - JSONL, uma linha por evento

O formato JSONL foi escolhido por ser consultavel por maquina sem parser
proprio. Depois da apresentacao, `jq` responde perguntas como "quantas consultas
foram retidas hoje?" direto no arquivo.

Cada consulta recebe um identificador. Sem ele, os eventos de consultas
simultaneas se misturariam no arquivo e a trilha deixaria de reconstruir o que
aconteceu em cada uma - que e justamente o proposito dela.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

ARQUIVO_PADRAO = "auditoria.jsonl"

# Icone por etapa. Numa demonstracao ao vivo, a coluna de icones deixa o
# caminho percorrido legivel sem que ninguem precise ler o texto.
ICONES = {
    "guardrail_entrada": "🛡",
    "consultar_prontuario": "🗄",
    "recuperar_evidencia": "📚",
    "responder": "🧠",
    "verificar_resposta": "⚕",
    "validacao_humana": "👤",
    "montar_resposta": "■",
}

CORES = {"INFO": "\033[36m", "ALERTA": "\033[33m", "CRITICO": "\033[31m"}
FIM = "\033[0m"


class TrilhaAuditoria:
    """
    Acumula os eventos de UMA consulta e os grava nos dois destinos.

    A instancia vive o tempo da consulta. Ela existe como objeto, e nao como
    funcoes soltas, porque o identificador da consulta precisa acompanhar todos
    os eventos - passa-lo a cada chamada seria mais uma coisa para esquecer.
    """

    def __init__(self, arquivo: str | Path = ARQUIVO_PADRAO, console: bool = True):
        self.trace_id = uuid.uuid4().hex[:12]
        self.arquivo = Path(arquivo)
        self.console = console
        self.eventos: list[dict] = []
        self._sequencia = 0

    def registrar(self, etapa: str, detalhe: str, duracao_ms: float,
                  nivel: str = "INFO", extra: dict | None = None) -> dict:
        self._sequencia += 1
        evento = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            "sequencia": self._sequencia,
            "nivel": nivel,
            "etapa": etapa,
            "detalhe": detalhe,
            "ms": round(duracao_ms, 2),
        }
        if extra:
            evento.update(extra)

        self.eventos.append(evento)
        self._gravar(evento)
        if self.console:
            self._imprimir(evento)
        return evento

    def _gravar(self, evento: dict) -> None:
        """Uma linha por evento, em JSON. Falha de escrita nao derruba a consulta."""
        try:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            with self.arquivo.open("a", encoding="utf-8") as saida:
                saida.write(json.dumps(evento, ensure_ascii=False) + "\n")
        except OSError as erro:
            # Perder o registro e ruim; derrubar o atendimento por causa disso e
            # pior. O aviso vai para stderr, que nao se confunde com a saida.
            print(f"[auditoria] nao foi possivel gravar: {erro}", file=sys.stderr)

    def _imprimir(self, evento: dict) -> None:
        cor = CORES.get(evento["nivel"], "")
        icone = ICONES.get(evento["etapa"], "·")
        print(
            f"{cor}[{evento['trace_id']}] {icone} {evento['etapa']:22} "
            f"{evento['ms']:8.2f} ms  {evento['detalhe']}{FIM}"
        )

    def resumo(self) -> dict:
        total = sum(e["ms"] for e in self.eventos)
        return {
            "trace_id": self.trace_id,
            "eventos": len(self.eventos),
            "duracao_total_ms": round(total, 2),
            "etapas": [e["etapa"] for e in self.eventos],
            "alertas": sum(1 for e in self.eventos if e["nivel"] != "INFO"),
        }


# A trilha da consulta em andamento.
#
# POR QUE CONTEXTVAR, E NAO UMA CHAVE DO ESTADO:
#     O estado do LangGraph e um TypedDict, e o framework DESCARTA em silencio
#     qualquer chave que nao esteja declarada nele. Um objeto passado por ali
#     simplesmente nao chega aos nos, e nada avisa - a consulta roda inteira e a
#     trilha sai vazia.
#
#     Declarar o objeto no estado resolveria o descarte e criaria outro
#     problema: o estado atravessa serializacao, e uma trilha com descritor de
#     arquivo aberto nao e serializavel.
#
#     Com contextvar, a trilha acompanha a consulta sem entrar no estado, e os
#     nos mantem a assinatura limpa `(estado) -> estado`.
_trilha_atual: ContextVar[TrilhaAuditoria | None] = ContextVar(
    "trilha_atual", default=None
)


def definir_trilha(trilha: TrilhaAuditoria | None) -> None:
    _trilha_atual.set(trilha)


def trilha_atual() -> TrilhaAuditoria | None:
    return _trilha_atual.get()


def configurar_logging(arquivo: str | Path = "medgraph.log",
                       nivel: int = logging.INFO) -> logging.Logger:
    """
    Logger de aplicacao, separado da trilha de auditoria.

    Os dois registram coisas diferentes: a trilha guarda o que aconteceu em cada
    CONSULTA, e este logger guarda o que aconteceu no SISTEMA - carga de modelo,
    construcao de indice, erro de infraestrutura. Misturar os dois tornaria a
    trilha ilegivel justamente quando ela e mais necessaria.
    """
    logger = logging.getLogger("medgraph")
    logger.setLevel(nivel)
    logger.handlers.clear()

    formato = logging.Formatter(
        "%(asctime)s  %(levelname)-8s %(name)s.%(funcName)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formato)
    logger.addHandler(console)

    try:
        em_arquivo = logging.FileHandler(arquivo, encoding="utf-8")
        em_arquivo.setFormatter(formato)
        logger.addHandler(em_arquivo)
    except OSError as erro:
        logger.warning("sem log em arquivo: %s", erro)

    return logger


def ler_trilha(arquivo: str | Path = ARQUIVO_PADRAO) -> list[dict]:
    """
    Le a trilha gravada. Linha invalida e ignorada, e nao derruba a leitura.

    O arquivo e escrito durante a execucao e pode terminar cortado se o processo
    for interrompido. Uma trilha parcial ainda e util; uma excecao de parse no
    meio da auditoria nao e.
    """
    caminho = Path(arquivo)
    if not caminho.exists():
        return []

    eventos = []
    with caminho.open(encoding="utf-8") as entrada:
        for linha in entrada:
            if not linha.strip():
                continue
            try:
                eventos.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return eventos


def consultas_registradas(arquivo: str | Path = ARQUIVO_PADRAO) -> dict[str, list[dict]]:
    """Agrupa os eventos por consulta, na ordem em que aconteceram."""
    por_consulta: dict[str, list[dict]] = {}
    for evento in ler_trilha(arquivo):
        por_consulta.setdefault(evento["trace_id"], []).append(evento)
    for eventos in por_consulta.values():
        eventos.sort(key=lambda e: e["sequencia"])
    return por_consulta
