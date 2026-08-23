"""
[REQ-3b] Configuracao de logging do MedGraph.

O QUE FAZ:
    Monta o sistema de log em TRES DESTINOS simultaneos, cada um com um
    proposito diferente:

      1. CONSOLE (colorido, via Rich)
         Para quem esta assistindo a execucao - a demonstracao em video e a
         revisao do professor. Mostra o passo a passo em tempo real.

      2. ARQUIVO logs/app.log (texto, rotativo)
         Historico legivel por humano, sobrevive ao fechamento do terminal.
         Roda em 5 arquivos de 5 MB para nao crescer sem controle.

      3. TRILHA DE AUDITORIA logs/auditoria/auditoria-AAAA-MM-DD.jsonl
         Um evento por linha, em JSON. Escrito pelo modulo `auditoria.py`,
         que usa a infraestrutura montada aqui. E este destino que atende
         literalmente ao "logging detalhado para rastreamento e auditoria"
         cobrado no item 3 do enunciado: cada linha e um registro imutavel,
         com carimbo de tempo, identificador de rastreio e dados estruturados
         prontos para serem consultados por ferramenta externa.

POR QUE TRES E NAO UM:
    Os tres publicos sao incompativeis. O console precisa ser bonito e
    resumido; o arquivo precisa ser completo e legivel; a auditoria precisa
    ser rigida e parseavel por maquina. Tentar servir aos tres com um
    formato so resulta em algo ruim para todos.

COMO USAR:
    from medgraph.logging_config import configurar_logging, obter_logger

    configurar_logging()                  # uma vez, no inicio do programa
    log = obter_logger(__name__)
    log.info("Indexando documentos...")
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import Settings, obter_settings

# Marca de controle: evita reconfigurar o logging se `configurar_logging()`
# for chamado mais de uma vez (comum quando varios modulos fazem bootstrap).
_JA_CONFIGURADO = False

# Nome do logger raiz do projeto. Todos os loggers do MedGraph descendem dele,
# o que permite ajustar o nivel do projeto inteiro sem mexer em bibliotecas
# de terceiros (langchain, httpx, urllib3 etc. costumam ser barulhentas).
LOGGER_RAIZ = "medgraph"

# Bibliotecas que poluem o console em nivel INFO. Sao rebaixadas para WARNING.
BIBLIOTECAS_SILENCIADAS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "sentence_transformers",
    "transformers",
    "datasets",
    "filelock",
    "matplotlib",
    "PIL",
)


class FormatadorJSONL(logging.Formatter):
    """
    Formata cada registro de log como uma unica linha JSON.

    POR QUE:
        A trilha de auditoria precisa ser consumida por maquina - seja pelo
        painel Streamlit (aba "Logs ao vivo"), seja por uma ferramenta de
        analise posterior. Texto livre exigiria parsing fragil por regex;
        JSON Lines resolve isso e ainda permite `jq` direto no terminal.

    CAMPOS FIXOS:
        ts        carimbo de tempo em UTC, formato ISO-8601
        nivel     INFO, WARNING, ERROR...
        logger    modulo que emitiu o registro
        mensagem  texto legivel
        Campos extras passados via `logger.info(..., extra={"dados": {...}})`
        sao anexados no mesmo objeto, achatados na raiz.
    """

    # Atributos internos do LogRecord que nao interessam a auditoria.
    _IGNORAR = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "nivel": record.levelname,
            "logger": record.name,
            "mensagem": record.getMessage(),
        }

        # Anexa qualquer campo extra passado pelo chamador.
        for chave, valor in record.__dict__.items():
            if chave in self._IGNORAR or chave.startswith("_"):
                continue
            payload[chave] = valor

        if record.exc_info:
            payload["excecao"] = self.formatException(record.exc_info)

        # default=str garante que objetos nao serializaveis (Path, datetime,
        # Decimal) virem texto em vez de derrubar a escrita do log.
        return json.dumps(payload, ensure_ascii=False, default=str)


class FiltroApenasAuditoria(logging.Filter):
    """
    Deixa passar somente registros marcados como eventos de auditoria.

    O modulo `auditoria.py` marca seus registros com `extra={"auditoria": True}`.
    Esse filtro garante que o arquivo JSONL contenha exclusivamente a trilha
    formal - e nao qualquer `log.debug()` espalhado pelo projeto.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "auditoria", False))


def _criar_handler_console(cfg: Settings) -> logging.Handler:
    """
    Handler do console. Usa Rich quando disponivel, senao cai para stderr puro.

    O fallback existe para o caso de o Rich nao estar instalado (ambiente
    minimo, CI). O projeto continua funcionando, so perde as cores.
    """
    if cfg.log_console_rich:
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_time=True,
                show_path=False,
                # markup=False de proposito: com markup ligado, o Rich
                # interpretaria colchetes que aparecem no CONTEUDO dos logs
                # (nomes de etapa, citacoes como [E1], trechos de prontuario)
                # como tags de estilo e os apagaria da tela. Num sistema de
                # auditoria, log que some e pior do que log sem cor.
                markup=False,
                log_time_format="[%H:%M:%S]",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler
        except ImportError:  # pragma: no cover - so ocorre sem o rich instalado
            pass

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


def _criar_handler_arquivo(cfg: Settings) -> logging.Handler:
    """Handler do logs/app.log, com rotacao em 5 arquivos de 5 MB."""
    cfg.dir_logs.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=cfg.dir_logs / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-32s | %(funcName)-24s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _criar_handler_auditoria(cfg: Settings) -> logging.Handler:
    """
    Handler da trilha de auditoria: um arquivo JSONL por dia.

    Optamos por particionar por data (e nao por tamanho) porque auditoria e
    consultada por periodo: "o que o assistente respondeu no dia 12?".
    """
    cfg.dir_auditoria.mkdir(parents=True, exist_ok=True)
    data_hoje = datetime.now().strftime("%Y-%m-%d")
    handler = logging.FileHandler(
        filename=cfg.dir_auditoria / f"auditoria-{data_hoje}.jsonl",
        encoding="utf-8",
    )
    handler.setFormatter(FormatadorJSONL())
    handler.addFilter(FiltroApenasAuditoria())
    # A trilha registra tudo, independente do LOG_LEVEL escolhido para o
    # console. Auditoria com nivel configuravel nao seria auditoria.
    handler.setLevel(logging.DEBUG)
    return handler


def configurar_logging(cfg: Settings | None = None, *, forcar: bool = False) -> None:
    """
    Monta os tres destinos de log. Deve ser chamada uma vez, no bootstrap.

    Args:
        cfg: configuracao a usar. Se omitida, usa `obter_settings()`.
        forcar: refaz a configuracao mesmo que ja tenha sido feita. Util em
            testes que precisam trocar o destino dos logs.
    """
    global _JA_CONFIGURADO
    if _JA_CONFIGURADO and not forcar:
        return

    cfg = cfg or obter_settings()

    logger_raiz = logging.getLogger(LOGGER_RAIZ)
    logger_raiz.setLevel(getattr(logging, cfg.log_level))
    # Impede que os registros subam para o root logger e sejam impressos duas
    # vezes caso alguma biblioteca tenha chamado logging.basicConfig().
    logger_raiz.propagate = False

    for handler in list(logger_raiz.handlers):
        logger_raiz.removeHandler(handler)
        handler.close()

    logger_raiz.addHandler(_criar_handler_console(cfg))

    if cfg.log_arquivo:
        logger_raiz.addHandler(_criar_handler_arquivo(cfg))

    if cfg.log_auditoria_jsonl:
        logger_raiz.addHandler(_criar_handler_auditoria(cfg))

    for nome in BIBLIOTECAS_SILENCIADAS:
        logging.getLogger(nome).setLevel(logging.WARNING)

    _JA_CONFIGURADO = True


def obter_logger(nome: str) -> logging.Logger:
    """
    Devolve um logger sob a hierarquia 'medgraph'.

    Aceita tanto `__name__` (ex.: "medgraph.rag.indexar", ja no lugar certo)
    quanto um nome solto (ex.: "indexacao", que vira "medgraph.indexacao").
    Isso evita que um modulo escape da configuracao por descuido de nome.
    """
    if nome == "__main__" or not nome.startswith(LOGGER_RAIZ):
        nome = f"{LOGGER_RAIZ}.{nome.replace('__main__', 'script')}"
    return logging.getLogger(nome)


def imprimir_banner(titulo: str, subtitulo: str = "", cfg: Settings | None = None) -> None:
    """
    Imprime um cabecalho visual no inicio de cada script.

    POR QUE ISSO EXISTE:
        O projeto sera apresentado em video. Um painel com o nome da etapa,
        o provedor de LLM ativo e o caminho dos logs deixa evidente, em uma
        olhada, o que esta rodando e onde procurar o rastro depois.
    """
    cfg = cfg or obter_settings()
    linha_config = (
        f"provider={cfg.llm_provider}  "
        f"modelo={cfg.ollama_model if cfg.llm_provider == 'ollama' else cfg.openai_model}  "
        f"embeddings={cfg.embedding_provider}"
    )

    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        corpo = f"[bold]{titulo}[/bold]"
        if subtitulo:
            corpo += f"\n[dim]{subtitulo}[/dim]"
        corpo += f"\n\n[cyan]{linha_config}[/cyan]"
        corpo += f"\n[dim]logs: {cfg.dir_logs}[/dim]"

        console.print(
            Panel(
                corpo,
                title=f"[bold green]{cfg.nome_projeto}[/bold green] v{cfg.versao}",
                subtitle="[dim]Tech Challenge Fase 3 - 8IADT[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
    except ImportError:  # pragma: no cover
        largura = 78
        print("=" * largura)
        print(f" {cfg.nome_projeto} v{cfg.versao} - {titulo}")
        if subtitulo:
            print(f" {subtitulo}")
        print(f" {linha_config}")
        print("=" * largura)


def caminho_auditoria_do_dia(cfg: Settings | None = None) -> Path:
    """Caminho do arquivo JSONL de auditoria referente a hoje."""
    cfg = cfg or obter_settings()
    return cfg.dir_auditoria / f"auditoria-{datetime.now():%Y-%m-%d}.jsonl"
