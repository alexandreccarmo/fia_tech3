"""
[REQ-3b] Trilha de auditoria do MedGraph.

O QUE FAZ:
    Registra, de forma estruturada e rastreavel, TUDO o que acontece durante
    o atendimento de uma pergunta clinica: qual no do grafo executou, quanto
    tempo levou, o que entrou, o que saiu, qual decisao foi tomada, qual
    guardrail aprovou ou reprovou, quais fontes foram citadas e quanto custou.

POR QUE EXISTE:
    O enunciado do Tech Challenge exige, no item 3, "implementar logging
    detalhado para rastreamento e auditoria". Em um contexto hospitalar isso
    nao e burocracia: se um assistente sugere uma conduta e algo da errado, e
    preciso reconstruir exatamente o que o sistema viu e por que respondeu
    daquele jeito. Um `print()` no meio do codigo nao permite isso.

OS TRES ARTEFATOS QUE ESTE MODULO PRODUZ:

    1. Eventos no console e em logs/app.log
       Acompanhamento em tempo real durante a demonstracao.

    2. logs/auditoria/auditoria-AAAA-MM-DD.jsonl
       Uma linha JSON por evento, de todas as consultas do dia. E a trilha
       formal, imutavel e consultavel por maquina.

    3. logs/traces/<trace_id>.json
       O dossie completo de UMA consulta: metadados, todos os eventos em
       ordem, tempos por etapa e o desfecho. E o arquivo que o painel
       Streamlit le para desenhar a aba "Trilha do grafo".

CONCEITO CENTRAL - O trace_id:
    Toda consulta ganha um identificador unico no momento em que entra no
    sistema. Esse identificador acompanha a requisicao por todos os nos do
    grafo, por todas as chamadas de LLM e por todas as consultas ao banco.
    Ele e propagado via `contextvars`, e nao por parametro, para que os nos
    do LangGraph continuem com assinatura limpa `(estado) -> estado`.

COMO USAR:

    from medgraph.auditoria import abrir_trilha, instrumentar, registrar

    with abrir_trilha(pergunta="Qual a conduta em sepse?", usuario="dr.silva") as t:
        resultado = grafo.invoke(estado)
        print(t.trace_id)

    @instrumentar("recuperar_evidencia", tipo=TipoEvento.RECUPERACAO)
    def recuperar_evidencia(estado): ...
"""

from __future__ import annotations

import functools
import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar

from config.settings import Settings, obter_settings
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Tamanho maximo, em caracteres, de um valor de texto gravado na trilha.
# Um abstract inteiro do PubMedQA tem ~1.500 caracteres; guardar todos eles
# em todos os eventos inflaria o JSONL sem ganho de auditabilidade.
LIMITE_TEXTO_EVENTO = 600

# Ate quantos itens uma lista pode ter para ser gravada por inteiro na trilha.
# Acima disso, vira um resumo {quantidade, amostra}.
MAX_ITENS_LISTA_INTEIRA = 12


class TipoEvento(StrEnum):
    """
    Vocabulario fechado dos tipos de evento da trilha.

    Usar um Enum (e nao strings livres) garante que consultas posteriores
    ao JSONL - do tipo "quantas vezes o guardrail reprovou este mes?" -
    funcionem sem depender de como cada desenvolvedor escreveu a palavra.
    """

    INICIO_TRILHA = "inicio_trilha"
    FIM_TRILHA = "fim_trilha"
    INICIO_ETAPA = "inicio_etapa"
    FIM_ETAPA = "fim_etapa"
    DECISAO = "decisao"            # roteamento condicional do grafo
    GUARDRAIL = "guardrail"        # aprovacao/reprovacao de politica  [REQ-3a]
    ANONIMIZACAO = "anonimizacao"  # PII removida                      [REQ-1a]
    LLM = "llm"                    # chamada ao modelo
    RECUPERACAO = "recuperacao"    # busca no indice vetorial          [REQ-3c]
    BANCO = "banco"                # consulta ao prontuario            [REQ-2a]
    REGRA_CLINICA = "regra_clinica"
    ALERTA = "alerta"              # alerta emitido a equipe medica
    VALIDACAO_HUMANA = "validacao_humana"  # human-in-the-loop         [REQ-3a]
    CUSTO = "custo"                # consumo de tokens / USD
    ERRO = "erro"


class Desfecho(StrEnum):
    """Como uma consulta terminou. Gravado no evento FIM_TRILHA."""

    RESPONDIDA = "respondida"
    RECUSADA = "recusada"                    # bloqueada pelo guardrail de entrada
    AGUARDANDO_VALIDACAO = "aguardando_validacao"
    DEGRADADA = "degradada"                  # guardrail de saida esgotou tentativas
    ERRO = "erro"


# -----------------------------------------------------------------------------
# REDACAO DE DADOS SENSIVEIS
# -----------------------------------------------------------------------------
# A trilha de auditoria NAO pode virar um vazamento de PII. Este gancho e
# substituido na Etapa 1 pela funcao real de anonimizacao
# (medgraph.dados.anonimizador.redigir). Ate la, e a identidade.
#
# Manter o gancho aqui, e nao importar o anonimizador diretamente, evita
# dependencia circular: o anonimizador tambem registra eventos de auditoria.
# -----------------------------------------------------------------------------
def _sem_redacao(texto: str) -> str:
    """Redator neutro, ativo ate que o anonimizador real seja instalado."""
    return texto


_redator: Callable[[str], str] = _sem_redacao


def definir_redator(funcao: Callable[[str], str]) -> None:
    """
    Instala a funcao de redacao de PII usada antes de gravar qualquer texto.

    Chamada uma vez no bootstrap, pelo modulo de anonimizacao. [REQ-1a]
    """
    global _redator
    _redator = funcao


def _resumir(valor: Any, *, limite: int = LIMITE_TEXTO_EVENTO) -> Any:
    """
    Reduz um valor a algo seguro e proporcional para gravar na trilha.

    REGRAS:
        - Texto longo e truncado, com marcacao explicita de quantos
          caracteres foram omitidos (a trilha nunca mente por omissao).
        - Listas CURTAS de valores curtos (ate 12 itens, cada um com no
          maximo 80 caracteres) sao mantidas inteiras: "quais chaves do
          estado" e "quais fontes foram citadas" e precisamente o que se
          quer poder auditar depois.
        - Listas grandes ou de itens volumosos viram {quantidade, amostra} -
          o que interessa e "recuperou 4 documentos", nao os 4 abstracts
          inteiros repetidos em cada evento.
        - Dicionarios sao percorridos recursivamente.
        - Objetos nao serializaveis viram sua representacao textual curta.
        - Todo texto passa pelo redator de PII antes de sair.
    """
    if valor is None or isinstance(valor, (bool, int, float)):
        return valor

    if isinstance(valor, str):
        texto = _redator(valor)
        if len(texto) <= limite:
            return texto
        return f"{texto[:limite]}... [+{len(texto) - limite} caracteres omitidos]"

    if isinstance(valor, dict):
        return {str(k): _resumir(v, limite=limite) for k, v in valor.items()}

    if isinstance(valor, (list, tuple, set)):
        itens = list(valor)
        cabe_inteira = len(itens) <= MAX_ITENS_LISTA_INTEIRA and all(
            isinstance(i, (str, int, float, bool)) and len(str(i)) <= 80 for i in itens
        )
        if cabe_inteira:
            return [_resumir(i, limite=limite) for i in itens]
        return {
            "quantidade": len(itens),
            "amostra": [_resumir(i, limite=200) for i in itens[:3]],
        }

    return _resumir(str(valor), limite=200)


@dataclass
class EventoAuditoria:
    """Um registro atomico da trilha. Corresponde a uma linha do JSONL."""

    tipo: TipoEvento
    mensagem: str
    trace_id: str
    sequencia: int
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    etapa: str | None = None
    duracao_ms: float | None = None
    dados: dict[str, Any] = field(default_factory=dict)
    nivel: str = "INFO"
    conclusao: bool = False
    """
    True apenas no evento que ENCERRA uma etapa.

    Marcar explicitamente (em vez de deduzir pelo tipo do evento) e o que
    permite que um no use um tipo proprio - GUARDRAIL, BANCO, LLM - sem
    desaparecer da linha do tempo reconstruida em `etapas_executadas()`.
    """

    def para_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tipo"] = self.tipo.value
        return d


@dataclass
class TrilhaAuditoria:
    """
    O dossie de uma consulta, do inicio ao fim.

    Acumula os eventos em memoria durante a execucao e grava o arquivo
    logs/traces/<trace_id>.json quando a consulta termina.
    """

    trace_id: str
    iniciada_em: str
    usuario: str | None = None
    paciente_id: str | None = None
    pergunta: str | None = None
    eventos: list[EventoAuditoria] = field(default_factory=list)
    configuracao: dict[str, Any] = field(default_factory=dict)
    desfecho: Desfecho | None = None
    finalizada_em: str | None = None
    duracao_total_ms: float | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    # -- registro ------------------------------------------------------------
    def registrar(
        self,
        tipo: TipoEvento,
        mensagem: str,
        *,
        etapa: str | None = None,
        duracao_ms: float | None = None,
        nivel: str = "INFO",
        conclusao: bool = False,
        **dados: Any,
    ) -> EventoAuditoria:
        """
        Anexa um evento a trilha e o publica nos destinos de log.

        O mesmo evento sai simultaneamente:
          - no console (legivel, com icone do tipo);
          - em logs/app.log;
          - em logs/auditoria/*.jsonl (por causa de extra={"auditoria": True}).
        """
        evento = EventoAuditoria(
            tipo=tipo,
            mensagem=mensagem,
            trace_id=self.trace_id,
            sequencia=len(self.eventos) + 1,
            etapa=etapa,
            duracao_ms=duracao_ms,
            dados={k: _resumir(v) for k, v in dados.items()},
            nivel=nivel,
            conclusao=conclusao,
        )
        self.eventos.append(evento)

        # Mensagem em texto puro: a mesma string vai para o console, para o
        # app.log e para a trilha JSONL. Estilo de terminal e responsabilidade
        # do handler, nunca do conteudo do registro de auditoria.
        sufixo = f" ({duracao_ms:.0f} ms)" if duracao_ms is not None else ""
        etiqueta = f"{etapa} · " if etapa else ""
        # Converte o nome do nivel ("INFO", "ERROR") no inteiro correspondente.
        # Cai para INFO se vier um nome desconhecido, em vez de derrubar a trilha.
        nivel_numerico = logging.getLevelName(nivel)
        if not isinstance(nivel_numerico, int):
            nivel_numerico = logging.INFO

        log.log(
            nivel_numerico,
            f"{_ICONES.get(tipo, '•')} {etiqueta}{mensagem}{sufixo}",
            extra={
                "auditoria": True,
                "trace_id": self.trace_id,
                "sequencia": evento.sequencia,
                "tipo": tipo.value,
                "etapa": etapa,
                "duracao_ms": duracao_ms,
                "conclusao": conclusao,
                "dados": evento.dados,
            },
        )
        return evento

    # -- consultas -----------------------------------------------------------
    def etapas_executadas(self) -> list[str]:
        """Nomes das etapas/nos percorridos, na ordem. Alimenta o painel visual."""
        vistas: list[str] = []
        for evento in self.eventos:
            if evento.conclusao and evento.etapa:
                vistas.append(evento.etapa)
        return vistas

    def tempo_por_etapa(self) -> dict[str, float]:
        """Latencia de cada etapa em milissegundos. Vira grafico no relatorio."""
        return {
            e.etapa: e.duracao_ms
            for e in self.eventos
            if e.conclusao and e.etapa and e.duracao_ms is not None
        }

    def eventos_do_tipo(self, tipo: TipoEvento) -> list[EventoAuditoria]:
        return [e for e in self.eventos if e.tipo is tipo]

    # -- persistencia --------------------------------------------------------
    def para_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "iniciada_em": self.iniciada_em,
            "finalizada_em": self.finalizada_em,
            "duracao_total_ms": self.duracao_total_ms,
            "usuario": self.usuario,
            "paciente_id": self.paciente_id,
            "pergunta": _resumir(self.pergunta),
            "desfecho": self.desfecho.value if self.desfecho else None,
            "configuracao": self.configuracao,
            "etapas_executadas": self.etapas_executadas(),
            "tempo_por_etapa_ms": self.tempo_por_etapa(),
            "total_eventos": len(self.eventos),
            "eventos": [e.para_dict() for e in self.eventos],
        }

    def salvar(self, cfg: Settings | None = None) -> None:
        """Grava logs/traces/<trace_id>.json com o dossie completo."""
        cfg = cfg or obter_settings()
        if not cfg.log_salvar_trace_completo:
            return
        cfg.dir_traces.mkdir(parents=True, exist_ok=True)
        destino = cfg.dir_traces / f"{self.trace_id}.json"
        destino.write_text(
            json.dumps(self.para_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


# Icones por tipo de evento - puramente visual, para a leitura no console
# durante a gravacao do video ficar imediata.
_ICONES: dict[TipoEvento, str] = {
    TipoEvento.INICIO_TRILHA: "▶",
    TipoEvento.FIM_TRILHA: "■",
    TipoEvento.INICIO_ETAPA: "→",
    TipoEvento.FIM_ETAPA: "✓",
    TipoEvento.DECISAO: "⑂",
    TipoEvento.GUARDRAIL: "🛡",
    TipoEvento.ANONIMIZACAO: "🔒",
    TipoEvento.LLM: "🧠",
    TipoEvento.RECUPERACAO: "📚",
    TipoEvento.BANCO: "🗄",
    TipoEvento.REGRA_CLINICA: "⚕",
    TipoEvento.ALERTA: "🚨",
    TipoEvento.VALIDACAO_HUMANA: "👤",
    TipoEvento.CUSTO: "💲",
    TipoEvento.ERRO: "✖",
}


# -----------------------------------------------------------------------------
# CONTEXTO AMBIENTE
# -----------------------------------------------------------------------------
# A trilha ativa fica em um ContextVar. Assim qualquer funcao, em qualquer
# profundidade da pilha, consegue registrar um evento sem que a trilha
# precise ser passada de parametro em parametro - o que manteria os nos do
# LangGraph com assinatura poluida.
# -----------------------------------------------------------------------------
_trilha_atual: ContextVar[TrilhaAuditoria | None] = ContextVar(
    "medgraph_trilha_atual", default=None
)


def trilha_atual() -> TrilhaAuditoria | None:
    """A trilha ativa neste contexto de execucao, ou None fora de uma consulta."""
    return _trilha_atual.get()


def registrar(tipo: TipoEvento, mensagem: str, **kwargs: Any) -> None:
    """
    Registra um evento na trilha ativa.

    Se nao houver trilha aberta (ex.: script de preparacao de dados rodando
    fora de uma consulta), o evento vai apenas para o log comum. Isso permite
    instrumentar codigo compartilhado sem exigir que todo chamador abra uma
    trilha.
    """
    trilha = trilha_atual()
    if trilha is not None:
        trilha.registrar(tipo, mensagem, **kwargs)
    else:
        etapa = kwargs.pop("etapa", None)
        etiqueta = f"{etapa} · " if etapa else ""
        log.info(f"{_ICONES.get(tipo, '•')} {etiqueta}{mensagem}")


@contextmanager
def abrir_trilha(
    *,
    pergunta: str | None = None,
    usuario: str | None = None,
    paciente_id: str | None = None,
    trace_id: str | None = None,
    cfg: Settings | None = None,
) -> Iterator[TrilhaAuditoria]:
    """
    Abre uma trilha de auditoria para uma consulta e garante o fechamento.

    Ao entrar, registra o evento INICIO_TRILHA com um retrato da configuracao
    vigente (sem segredos). Ao sair, registra FIM_TRILHA com a duracao total
    e o desfecho, e grava o arquivo de trace.

    Em caso de excecao, marca o desfecho como ERRO, registra o evento e
    RE-LEVANTA a excecao - auditoria nunca engole erro.
    """
    cfg = cfg or obter_settings()
    trilha = TrilhaAuditoria(
        trace_id=trace_id or uuid.uuid4().hex[:16],
        iniciada_em=datetime.now(UTC).isoformat(),
        usuario=usuario,
        paciente_id=paciente_id,
        pergunta=pergunta,
        configuracao=cfg.resumo_seguro(),
    )
    token: Token = _trilha_atual.set(trilha)

    trilha.registrar(
        TipoEvento.INICIO_TRILHA,
        "Consulta recebida",
        usuario=usuario,
        paciente_id=paciente_id,
        pergunta=pergunta,
    )

    try:
        yield trilha
    except Exception as exc:
        trilha.desfecho = Desfecho.ERRO
        trilha.registrar(
            TipoEvento.ERRO,
            f"Falha nao tratada: {type(exc).__name__}: {exc}",
            nivel="ERROR",
            excecao=type(exc).__name__,
        )
        raise
    finally:
        trilha.duracao_total_ms = (time.perf_counter() - trilha._t0) * 1000
        trilha.finalizada_em = datetime.now(UTC).isoformat()
        if trilha.desfecho is None:
            trilha.desfecho = Desfecho.RESPONDIDA
        trilha.registrar(
            TipoEvento.FIM_TRILHA,
            f"Consulta finalizada ({trilha.desfecho.value})",
            duracao_ms=trilha.duracao_total_ms,
            etapas=trilha.etapas_executadas(),
            total_eventos=len(trilha.eventos),
        )
        trilha.salvar(cfg)
        _trilha_atual.reset(token)


@contextmanager
def etapa(nome: str, tipo: TipoEvento = TipoEvento.FIM_ETAPA, **dados: Any) -> Iterator[None]:
    """
    Cronometra e audita um bloco de codigo que nao e uma funcao inteira.

    Uso:
        with etapa("carregar_indice_faiss", caminho=str(p)):
            vectorstore = FAISS.load_local(...)
    """
    registrar(TipoEvento.INICIO_ETAPA, "Iniciando", etapa=nome, **dados)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        registrar(
            TipoEvento.ERRO,
            f"Erro: {type(exc).__name__}: {exc}",
            etapa=nome,
            duracao_ms=(time.perf_counter() - t0) * 1000,
            nivel="ERROR",
        )
        raise
    else:
        registrar(
            tipo,
            "Concluido",
            etapa=nome,
            duracao_ms=(time.perf_counter() - t0) * 1000,
            conclusao=True,
        )


def instrumentar(
    nome: str | None = None,
    *,
    tipo: TipoEvento = TipoEvento.FIM_ETAPA,
    registrar_entrada: bool = True,
    registrar_saida: bool = True,
) -> Callable[[F], F]:
    """
    [REQ-3b] Decorator que audita automaticamente um no do grafo.

    O QUE FAZ:
        Envolve a funcao decorada para que, sem nenhuma linha extra dentro
        dela, sejam registrados: inicio, chaves recebidas no estado, chaves
        devolvidas, tempo de execucao e eventuais excecoes.

    POR QUE UM DECORATOR:
        A alternativa seria repetir o mesmo bloco de log no comeco e no fim
        dos 12 nos do grafo. Alem de verboso, alguem esqueceria em algum no -
        e um no sem rastro derruba a garantia de auditabilidade que o
        requisito 3 exige. Com o decorator, auditar passa a ser o padrao.

    O QUE E GRAVADO DO ESTADO:
        Apenas as CHAVES presentes na entrada e o CONTEUDO RESUMIDO do delta
        devolvido. O estado clinico completo circula com dados do paciente;
        grava-lo inteiro a cada no encheria a trilha de PII repetida.

    Args:
        nome: nome da etapa na trilha. Se omitido, usa o nome da funcao.
        tipo: tipo de evento do registro de conclusao.
        registrar_entrada: grava as chaves do estado de entrada.
        registrar_saida: grava o delta devolvido pelo no.
    """

    def decorador(funcao: F) -> F:
        etiqueta = nome or funcao.__name__

        @functools.wraps(funcao)
        def envolvida(*args: Any, **kwargs: Any) -> Any:
            entrada: dict[str, Any] = {}
            if registrar_entrada and args and isinstance(args[0], dict):
                entrada["chaves_estado"] = sorted(args[0].keys())

            registrar(TipoEvento.INICIO_ETAPA, "Iniciando", etapa=etiqueta, **entrada)
            t0 = time.perf_counter()

            try:
                resultado = funcao(*args, **kwargs)
            except Exception as exc:
                registrar(
                    TipoEvento.ERRO,
                    f"Erro: {type(exc).__name__}: {exc}",
                    etapa=etiqueta,
                    duracao_ms=(time.perf_counter() - t0) * 1000,
                    nivel="ERROR",
                    excecao=type(exc).__name__,
                )
                raise

            saida: dict[str, Any] = {}
            if registrar_saida and isinstance(resultado, dict):
                saida["delta"] = resultado

            registrar(
                tipo,
                "Concluido",
                etapa=etiqueta,
                duracao_ms=(time.perf_counter() - t0) * 1000,
                conclusao=True,
                **saida,
            )
            return resultado

        return envolvida  # type: ignore[return-value]

    return decorador
