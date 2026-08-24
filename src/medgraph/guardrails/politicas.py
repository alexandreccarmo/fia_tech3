"""
[REQ-3a] Carregamento das políticas declarativas.

O QUE FAZ:
    Lê `config/politicas.yaml`, compila as expressões regulares uma única vez
    e expõe as regras como objetos consultáveis.

POR QUE COMPILAR NA CARGA, E NÃO A CADA USO:
    As regexes são aplicadas em todas as consultas. Compilá-las a cada
    chamada custaria tempo em um caminho quente. Mais importante: compilar na
    carga faz uma regex inválida falhar no bootstrap da aplicação, e não no
    meio de um atendimento — que é quando o guardrail é chamado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from re import Pattern
from typing import Any

import yaml

from config.settings import Settings, obter_settings
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


@dataclass(frozen=True)
class PadraoBloqueio:
    """Um padrão que, se encontrado na entrada, provoca recusa imediata."""

    id: str
    regex: Pattern[str]
    motivo: str


@dataclass
class Politicas:
    """As regras de atuação do assistente, já compiladas."""

    bruto: dict[str, Any]
    padroes_bloqueio: list[PadraoBloqueio] = field(default_factory=list)
    padroes_posologia: list[Pattern[str]] = field(default_factory=list)
    formato_citacao: Pattern[str] | None = None
    termos_emergencia: list[str] = field(default_factory=list)

    # -- acesso conveniente às seções ---------------------------------------
    @property
    def identidade(self) -> dict[str, Any]:
        return self.bruto.get("identidade", {})

    @property
    def escopo(self) -> dict[str, Any]:
        return self.bruto.get("escopo", {})

    @property
    def entrada(self) -> dict[str, Any]:
        return self.bruto.get("entrada", {})

    @property
    def saida(self) -> dict[str, Any]:
        return self.bruto.get("saida", {})

    @property
    def risco(self) -> dict[str, Any]:
        return self.bruto.get("risco", {})

    def texto(self, chave: str) -> str:
        """Um dos textos padronizados. Levanta KeyError se a chave não existir."""
        textos = self.bruto.get("textos", {})
        if chave not in textos:
            raise KeyError(
                f"Texto '{chave}' ausente de politicas.yaml. Disponíveis: {sorted(textos)}"
            )
        return " ".join(str(textos[chave]).split())

    def intencoes_permitidas(self) -> list[str]:
        return [i["id"] for i in self.escopo.get("intencoes_permitidas", [])]

    def intencao(self, identificador: str) -> dict[str, Any] | None:
        for item in self.escopo.get("intencoes_permitidas", []):
            if item["id"] == identificador:
                return item
        return None

    def peso_risco(self, gatilho: str) -> float:
        for item in self.risco.get("gatilhos", []):
            if item["id"] == gatilho:
                return float(item["peso"])
        return 0.0


@lru_cache(maxsize=1)
def carregar(caminho: str | None = None) -> Politicas:
    """
    Carrega e compila o arquivo de políticas.

    Cacheado: o arquivo é lido uma vez por processo. Em testes que alterem o
    YAML, chame `carregar.cache_clear()`.
    """
    cfg: Settings = obter_settings()
    origem = caminho or str(cfg.caminho_politicas)

    with open(origem, encoding="utf-8") as arquivo:
        bruto = yaml.safe_load(arquivo)

    politicas = Politicas(bruto=bruto)

    for item in bruto.get("entrada", {}).get("padroes_bloqueio", []):
        try:
            politicas.padroes_bloqueio.append(
                PadraoBloqueio(
                    id=item["id"],
                    regex=re.compile(item["regex"], re.IGNORECASE),
                    motivo=item["motivo"],
                )
            )
        except re.error as exc:
            raise ValueError(
                f"Regex inválida na política '{item['id']}' de {origem}: {exc}"
            ) from exc

    for padrao in bruto.get("saida", {}).get("padroes_posologia", []):
        politicas.padroes_posologia.append(re.compile(padrao, re.IGNORECASE))

    formato = bruto.get("saida", {}).get("formato_citacao")
    if formato:
        politicas.formato_citacao = re.compile(formato)

    politicas.termos_emergencia = [
        t.lower() for t in bruto.get("entrada", {}).get("termos_emergencia", [])
    ]

    log.debug(
        "Políticas versão %s carregadas: %d padrão(ões) de bloqueio, %d de posologia",
        bruto.get("versao"),
        len(politicas.padroes_bloqueio),
        len(politicas.padroes_posologia),
    )
    return politicas
