"""
[REQ-3b] Contabilidade de consumo e trava de orcamento.

O QUE FAZ:
    Contabiliza cada chamada a um modelo de linguagem - quantos tokens
    entraram, quantos sairam, quanto custou em dolares - e impede que a
    execucao ultrapasse o teto definido em MAX_CUSTO_USD_SESSAO.

POR QUE EXISTE:
    Duas razoes, uma pratica e uma academica.

    PRATICA: o projeto usa a OpenAI apenas como teto de referencia na
    avaliacao. Um laco mal escrito ou um retry infinito poderia consumir
    credito real da conta. A trava corta o problema na origem: ao atingir o
    limite, o provider recusa a chamada com erro explicito em vez de
    continuar gastando silenciosamente.

    ACADEMICA: "logging detalhado para rastreamento e auditoria" (item 3 do
    enunciado) inclui saber o custo do que foi executado. A tabela de consumo
    gerada por este modulo entra no relatorio tecnico e demonstra a diferenca
    economica entre o modelo fine-tunado local (custo zero por consulta) e a
    API paga.

MODELO DE COBRANCA:
    A OpenAI cobra por token, com precos diferentes para entrada e saida.
    Os valores em PRECOS_USD_POR_MILHAO sao um retrato local usado para
    ESTIMATIVA - a fatura real e sempre a da plataforma. Sao conferiveis em
    https://openai.com/api/pricing e devem ser revisados se o projeto for
    retomado muito depois.

COMO USAR:
    from medgraph.llm.custo import contador

    c = contador()
    c.registrar_uso("gpt-4o-mini", tokens_entrada=1200, tokens_saida=300)
    print(c.total_usd)          # 0.00036
    c.verificar_orcamento()     # levanta OrcamentoExcedidoError se estourou
    print(c.tabela_resumo())
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


# -----------------------------------------------------------------------------
# TABELA DE PRECOS
# -----------------------------------------------------------------------------
# Valores em dolares por MILHAO de tokens, no formato (entrada, saida).
# Modelos locais aparecem com custo zero de proposito: eles APARECEM na
# contabilidade (para que o relatorio mostre o volume processado), mas nao
# somam gasto financeiro.
# -----------------------------------------------------------------------------
PRECOS_USD_POR_MILHAO: Final[dict[str, tuple[float, float]]] = {
    # --- OpenAI: geracao ---
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # --- OpenAI: embeddings (cobram somente entrada) ---
    "text-embedding-3-small": (0.02, 0.00),
    "text-embedding-3-large": (0.13, 0.00),
    # --- Modelos locais: sem custo financeiro ---
    "local": (0.00, 0.00),
    "eco": (0.00, 0.00),
}

# Provedores que rodam na própria máquina. Qualquer modelo servido por eles
# tem custo financeiro zero, independentemente do nome que receba.
#
# Identificar pelo PROVEDOR, e não por uma lista de nomes, é o que torna isso
# correto: o nome do modelo local vem do .env e pode ser qualquer coisa
# ("medgraph", "medgraph-base", "medgraph-v2"). Uma lista de nomes deixaria
# de fora o primeiro nome novo que alguém escolhesse — e o contador atribuiria
# a ele o preço da OpenAI, inflando a tabela de custo do relatório com um
# gasto que nunca existiu.
PROVEDORES_LOCAIS: Final[frozenset[str]] = frozenset({"ollama", "eco", "local"})

# Preco assumido para um modelo desconhecido. Escolhemos o do gpt-4o-mini
# (o mais provavel neste projeto) e registramos um aviso, em vez de assumir
# zero - subestimar custo e pior do que superestimar.
PRECO_PADRAO: Final[tuple[float, float]] = PRECOS_USD_POR_MILHAO["gpt-4o-mini"]


class OrcamentoExcedidoError(RuntimeError):
    """
    Levantada quando uma chamada paga ultrapassaria o teto da sessao.

    E um erro, e nao um aviso, de proposito: o objetivo e INTERROMPER o
    processamento, nao continuar gastando com um log de alerta que ninguem le.
    """


@dataclass(frozen=True)
class RegistroUso:
    """Uma chamada de modelo contabilizada."""

    ts: str
    modelo: str
    tokens_entrada: int
    tokens_saida: int
    custo_usd: float
    origem: str
    """Onde no projeto a chamada nasceu. Ex.: 'avaliacao', 'grafo.raciocinio_clinico'."""

    provedor: str = ""
    """ollama, openai ou eco. Determina se a chamada tem custo financeiro."""


@dataclass
class ContadorCusto:
    """
    Acumulador de consumo de uma sessao de execucao.

    Uma "sessao" e um processo Python. Cada `python scripts/04_avaliar.py`
    comeca do zero; o teto de MAX_CUSTO_USD_SESSAO vale para aquela execucao.
    """

    limite_usd: float
    registros: list[RegistroUso] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- calculo -------------------------------------------------------------
    @staticmethod
    def calcular_custo(
        modelo: str,
        tokens_entrada: int,
        tokens_saida: int,
        *,
        provedor: str | None = None,
    ) -> float:
        """
        Custo em USD de uma chamada, segundo a tabela de precos local.

        Args:
            provedor: quando informado e local (ollama/eco), o custo e zero
                qualquer que seja o nome do modelo.
        """
        if provedor and provedor.lower() in PROVEDORES_LOCAIS:
            return 0.0

        preco = PRECOS_USD_POR_MILHAO.get(modelo)
        if preco is None:
            log.warning(
                "Modelo %r nao esta na tabela de precos; estimando com o preco do "
                "gpt-4o-mini. Atualize PRECOS_USD_POR_MILHAO se o valor importar.",
                modelo,
            )
            preco = PRECO_PADRAO
        entrada_usd = tokens_entrada / 1_000_000 * preco[0]
        saida_usd = tokens_saida / 1_000_000 * preco[1]
        return entrada_usd + saida_usd

    # -- registro ------------------------------------------------------------
    def registrar_uso(
        self,
        modelo: str,
        tokens_entrada: int,
        tokens_saida: int = 0,
        *,
        origem: str = "nao_informada",
        provedor: str | None = None,
    ) -> RegistroUso:
        """
        Contabiliza uma chamada ja realizada e publica o evento na auditoria.

        Repare que este metodo registra o que JA aconteceu. A decisao de
        permitir ou barrar uma chamada e de `verificar_orcamento()`, que deve
        ser consultado ANTES de chamar o modelo.
        """
        custo = self.calcular_custo(modelo, tokens_entrada, tokens_saida, provedor=provedor)
        registro = RegistroUso(
            ts=datetime.now(UTC).isoformat(),
            modelo=modelo,
            tokens_entrada=tokens_entrada,
            tokens_saida=tokens_saida,
            custo_usd=custo,
            origem=origem,
            provedor=provedor or "",
        )
        with self._lock:
            self.registros.append(registro)

        registrar(
            TipoEvento.CUSTO,
            f"{modelo}: {tokens_entrada} entrada + {tokens_saida} saida = US$ {custo:.6f}",
            modelo=modelo,
            tokens_entrada=tokens_entrada,
            tokens_saida=tokens_saida,
            custo_usd=round(custo, 6),
            acumulado_usd=round(self.total_usd, 6),
            origem=origem,
        )
        return registro

    # -- totais --------------------------------------------------------------
    @property
    def total_usd(self) -> float:
        return sum(r.custo_usd for r in self.registros)

    @property
    def total_tokens_entrada(self) -> int:
        return sum(r.tokens_entrada for r in self.registros)

    @property
    def total_tokens_saida(self) -> int:
        return sum(r.tokens_saida for r in self.registros)

    @property
    def total_chamadas(self) -> int:
        return len(self.registros)

    @property
    def saldo_usd(self) -> float:
        """Quanto ainda pode ser gasto nesta sessao antes de bater o teto."""
        return max(0.0, self.limite_usd - self.total_usd)

    # -- trava ---------------------------------------------------------------
    def verificar_orcamento(self, custo_previsto_usd: float = 0.0) -> None:
        """
        Autoriza (ou nao) uma proxima chamada paga.

        Deve ser consultado ANTES de chamar o modelo. Se o gasto acumulado
        somado ao custo previsto ultrapassar o teto, levanta
        OrcamentoExcedidoError com uma mensagem que explica exatamente o que
        fazer - aumentar MAX_CUSTO_USD_SESSAO no .env ou trocar de provider.
        """
        projetado = self.total_usd + custo_previsto_usd
        if projetado > self.limite_usd:
            registrar(
                TipoEvento.CUSTO,
                "Chamada bloqueada: orcamento da sessao esgotado",
                nivel="ERROR",
                gasto_usd=round(self.total_usd, 6),
                limite_usd=self.limite_usd,
                projetado_usd=round(projetado, 6),
            )
            raise OrcamentoExcedidoError(
                f"Orcamento da sessao esgotado: ja foram gastos US$ {self.total_usd:.4f} "
                f"de um teto de US$ {self.limite_usd:.2f} "
                f"(esta chamada custaria mais US$ {custo_previsto_usd:.4f}).\n"
                f"Para continuar: aumente MAX_CUSTO_USD_SESSAO no .env, "
                f"ou use LLM_PROVIDER=ollama (modelo local, sem custo)."
            )

    # -- relatorio -----------------------------------------------------------
    def por_modelo(self) -> dict[str, dict[str, float | int]]:
        """Consumo agregado por modelo. Base da tabela do relatorio tecnico."""
        agregado: dict[str, dict[str, float | int]] = {}
        for r in self.registros:
            linha = agregado.setdefault(
                r.modelo,
                {"chamadas": 0, "tokens_entrada": 0, "tokens_saida": 0, "custo_usd": 0.0},
            )
            linha["chamadas"] = int(linha["chamadas"]) + 1
            linha["tokens_entrada"] = int(linha["tokens_entrada"]) + r.tokens_entrada
            linha["tokens_saida"] = int(linha["tokens_saida"]) + r.tokens_saida
            linha["custo_usd"] = float(linha["custo_usd"]) + r.custo_usd
        return agregado

    def tabela_resumo(self) -> str:
        """Resumo em texto puro, pronto para colar no console ou no relatorio."""
        if not self.registros:
            return "Nenhuma chamada de modelo contabilizada nesta sessao."

        linhas = [
            f"{'Modelo':<28} {'Chamadas':>9} {'Tok.entrada':>12} {'Tok.saida':>10} {'US$':>10}",
            "-" * 73,
        ]
        for modelo, d in sorted(self.por_modelo().items()):
            linhas.append(
                f"{modelo:<28} {int(d['chamadas']):>9} {int(d['tokens_entrada']):>12} "
                f"{int(d['tokens_saida']):>10} {float(d['custo_usd']):>10.6f}"
            )
        linhas.append("-" * 73)
        linhas.append(
            f"{'TOTAL':<28} {self.total_chamadas:>9} {self.total_tokens_entrada:>12} "
            f"{self.total_tokens_saida:>10} {self.total_usd:>10.6f}"
        )
        linhas.append(
            f"Teto da sessao: US$ {self.limite_usd:.2f}  |  Saldo: US$ {self.saldo_usd:.4f}"
        )
        return "\n".join(linhas)

    def salvar(self, cfg: Settings | None = None) -> None:
        """
        Anexa os registros da sessao em logs/custos.jsonl.

        Arquivo separado da trilha de auditoria porque tem outro ciclo de
        vida: a auditoria e por consulta, o custo e acumulado ao longo do
        projeto inteiro e alimenta a tabela final do relatorio.
        """
        cfg = cfg or obter_settings()
        if not self.registros:
            return
        cfg.dir_logs.mkdir(parents=True, exist_ok=True)
        destino = cfg.dir_logs / "custos.jsonl"
        with destino.open("a", encoding="utf-8") as arquivo:
            for r in self.registros:
                arquivo.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# INSTANCIA DE SESSAO
# -----------------------------------------------------------------------------
_contador: ContadorCusto | None = None
_lock_criacao = threading.Lock()


def contador(cfg: Settings | None = None) -> ContadorCusto:
    """
    Devolve o contador unico da sessao, criando-o na primeira chamada.

    Ser um singleton e essencial: se cada modulo tivesse seu proprio contador,
    o teto de orcamento seria multiplicado pelo numero de modulos.
    """
    global _contador
    if _contador is None:
        with _lock_criacao:
            if _contador is None:
                cfg = cfg or obter_settings()
                _contador = ContadorCusto(limite_usd=cfg.max_custo_usd_sessao)
    return _contador


def reiniciar_contador(limite_usd: float | None = None) -> ContadorCusto:
    """Zera a contabilidade. Usado pelos testes e por scripts de longa duracao."""
    global _contador
    with _lock_criacao:
        limite = limite_usd if limite_usd is not None else obter_settings().max_custo_usd_sessao
        _contador = ContadorCusto(limite_usd=limite)
    return _contador


def estimar_tokens(texto: str, modelo: str = "gpt-4o-mini") -> int:
    """
    Estima quantos tokens um texto consome.

    Usa tiktoken quando disponivel (contagem exata para modelos OpenAI). Sem
    tiktoken, cai para a heuristica de ~4 caracteres por token, que erra
    pouco em portugues e ingles e serve bem para uma trava de orcamento.

    A estimativa e usada ANTES da chamada, para consultar o orcamento; o
    numero real de tokens vem depois, na resposta da API.
    """
    try:
        import tiktoken

        try:
            codificador = tiktoken.encoding_for_model(modelo)
        except KeyError:
            codificador = tiktoken.get_encoding("o200k_base")
        return len(codificador.encode(texto))
    except Exception:  # pragma: no cover - fallback sem tiktoken instalado
        return max(1, len(texto) // 4)
