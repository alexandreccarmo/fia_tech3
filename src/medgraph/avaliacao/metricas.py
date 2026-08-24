"""
[REQ-E3] Métricas de avaliação do modelo.

O QUE FAZ:
    Extrai o rótulo previsto do texto gerado pelo modelo e calcula as métricas
    que sustentam a seção de avaliação do relatório técnico.

POR QUE MACRO-F1 E NÃO SÓ ACCURACY:
    O conjunto de teste do PubMedQA tem 276 "yes", 169 "no" e 55 "maybe". Um
    modelo que responda "yes" para tudo acerta 55,2% — e é completamente
    inútil. A accuracy sozinha não denuncia isso.

    O macro-F1 calcula o F1 de cada classe separadamente e tira a média
    simples. A classe "maybe", com 11% dos casos, pesa tanto quanto "yes".
    Um modelo que nunca preveja "maybe" tem F1 zero nessa classe e macro-F1
    limitado a cerca de 0,67, por mais que a accuracy pareça boa.

    É por isso que as duas métricas aparecem lado a lado em todos os
    resultados: a accuracy é o número intuitivo, o macro-F1 é o número
    honesto.

A TAXA DE EXTRAÇÃO É UMA MÉTRICA, NÃO UM DETALHE:
    Se o modelo não produzir a linha "Decisão: ...", não há rótulo para
    comparar. Tratar isso como erro de predição misturaria dois problemas
    diferentes — errar a resposta e não seguir o formato. Contamos os dois
    separadamente, e a taxa de adesão ao formato entra no relatório: ela é a
    evidência direta do efeito do fine-tuning sobre o comportamento do modelo.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

ROTULOS = ("yes", "no", "maybe")

# A primeira linha da resposta deve ser "Decisão: <rótulo>". Aceitamos com e
# sem acento, em qualquer caixa, com ou sem os dois-pontos.
PADRAO_DECISAO = re.compile(
    r"decis[ãa]o\s*:?\s*\**\s*(yes|no|maybe|sim|n[ãa]o|talvez)", re.IGNORECASE
)

# Tradução dos rótulos em português que o modelo às vezes produz apesar da
# instrução. Aceitá-los é a decisão certa: o objetivo da métrica é medir o
# raciocínio clínico, não punir a escolha do idioma na palavra do rótulo.
TRADUCAO = {"sim": "yes", "não": "no", "nao": "no", "talvez": "maybe"}


def extrair_decisao(texto: str) -> tuple[str | None, str]:
    """
    Recupera o rótulo previsto a partir da resposta em texto livre.

    Returns:
        (rótulo, método) — o rótulo em minúsculas ou None, e como foi obtido:
        "formato" quando veio da linha "Decisão:", "busca" quando foi
        localizado em outro ponto do texto, "ausente" quando não há rótulo.

        Distinguir "formato" de "busca" importa: a proporção de respostas que
        chegam pelo caminho "formato" mede a aderência do modelo ao contrato
        de saída, que é um dos efeitos esperados do fine-tuning.
    """
    if not texto:
        return None, "ausente"

    correspondencia = PADRAO_DECISAO.search(texto)
    if correspondencia:
        bruto = correspondencia.group(1).lower()
        return TRADUCAO.get(bruto, bruto), "formato"

    # Recurso secundário: o rótulo pode aparecer solto no texto. Procuramos
    # nas primeiras 200 posições, onde uma resposta direta o colocaria.
    inicio = texto[:200].lower()
    for rotulo in ROTULOS:
        if re.search(rf"\b{rotulo}\b", inicio):
            return rotulo, "busca"

    return None, "ausente"


@dataclass
class ResultadoAvaliacao:
    """Resultado completo da avaliação de um sistema sobre o conjunto de teste."""

    sistema: str
    verdadeiros: list[str] = field(default_factory=list)
    previstos: list[str | None] = field(default_factory=list)
    metodos: list[str] = field(default_factory=list)
    latencias_ms: list[float] = field(default_factory=list)
    respostas: list[str] = field(default_factory=list)
    perguntas: list[str] = field(default_factory=list)
    erros: int = 0

    # -- contagens ----------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.verdadeiros)

    @property
    def extraidos(self) -> int:
        return sum(1 for p in self.previstos if p is not None)

    @property
    def taxa_extracao(self) -> float:
        return self.extraidos / self.total if self.total else 0.0

    @property
    def taxa_adesao_formato(self) -> float:
        """Proporção de respostas que trouxeram a linha 'Decisão:' esperada."""
        if not self.total:
            return 0.0
        return sum(1 for m in self.metodos if m == "formato") / self.total

    # -- métricas -----------------------------------------------------------
    def _pares_validos(self) -> tuple[list[str], list[str]]:
        """
        Pares (verdadeiro, previsto) em que houve rótulo extraído.

        Respostas sem rótulo são contabilizadas à parte, na taxa de extração.
        Somá-las como erro de classificação misturaria "errou a resposta" com
        "não seguiu o formato", que são problemas distintos e têm correções
        distintas.
        """
        pares = [
            (v, p) for v, p in zip(self.verdadeiros, self.previstos, strict=True) if p is not None
        ]
        if not pares:
            return [], []
        verdadeiros, previstos = zip(*pares, strict=True)
        return list(verdadeiros), list(previstos)

    @property
    def accuracy(self) -> float:
        """
        Accuracy sobre TODO o conjunto.

        Respostas sem rótulo extraído contam como erro. É o número correto do
        ponto de vista de quem usa o sistema: uma resposta ilegível não serve,
        mesmo que o raciocínio por trás estivesse certo.
        """
        if not self.total:
            return 0.0
        acertos = sum(
            1 for v, p in zip(self.verdadeiros, self.previstos, strict=True) if p == v
        )
        return acertos / self.total

    @property
    def accuracy_entre_extraidos(self) -> float:
        """Accuracy considerando apenas as respostas em que houve rótulo."""
        verdadeiros, previstos = self._pares_validos()
        if not verdadeiros:
            return 0.0
        return sum(1 for v, p in zip(verdadeiros, previstos, strict=True) if v == p) / len(verdadeiros)

    def f1_por_classe(self) -> dict[str, dict[str, float]]:
        """Precisão, revocação, F1 e suporte de cada classe."""
        resultado: dict[str, dict[str, float]] = {}
        for rotulo in ROTULOS:
            vp = sum(
                1 for v, p in zip(self.verdadeiros, self.previstos, strict=True)
                if v == rotulo and p == rotulo
            )
            fp = sum(
                1 for v, p in zip(self.verdadeiros, self.previstos, strict=True)
                if v != rotulo and p == rotulo
            )
            fn = sum(
                1 for v, p in zip(self.verdadeiros, self.previstos, strict=True)
                if v == rotulo and p != rotulo
            )
            precisao = vp / (vp + fp) if (vp + fp) else 0.0
            revocacao = vp / (vp + fn) if (vp + fn) else 0.0
            f1 = (
                2 * precisao * revocacao / (precisao + revocacao)
                if (precisao + revocacao)
                else 0.0
            )
            resultado[rotulo] = {
                "precisao": precisao,
                "revocacao": revocacao,
                "f1": f1,
                "suporte": sum(1 for v in self.verdadeiros if v == rotulo),
            }
        return resultado

    @property
    def macro_f1(self) -> float:
        """Média simples do F1 das três classes. A métrica principal."""
        por_classe = self.f1_por_classe()
        return sum(d["f1"] for d in por_classe.values()) / len(ROTULOS)

    def matriz_confusao(self) -> dict[str, dict[str, int]]:
        """
        Matriz verdadeiro × previsto, com a coluna extra "(sem rótulo)".

        A coluna extra é o que revela um modo de falha que a matriz 3x3
        esconderia: o modelo pode estar deixando de responder em vez de
        errando.
        """
        colunas = [*ROTULOS, "(sem rótulo)"]
        matriz = {v: dict.fromkeys(colunas, 0) for v in ROTULOS}
        for verdadeiro, previsto in zip(self.verdadeiros, self.previstos, strict=True):
            matriz[verdadeiro][previsto if previsto in ROTULOS else "(sem rótulo)"] += 1
        return matriz

    @property
    def distribuicao_previsoes(self) -> dict[str, int]:
        """
        Como o modelo distribuiu suas respostas.

        Delata na hora o modelo que colapsou numa classe só — o modo de falha
        mais comum e mais fácil de confundir com bom desempenho.
        """
        return dict(Counter(p or "(sem rótulo)" for p in self.previstos))

    @property
    def latencia_media_ms(self) -> float:
        return sum(self.latencias_ms) / len(self.latencias_ms) if self.latencias_ms else 0.0

    @property
    def latencia_mediana_ms(self) -> float:
        if not self.latencias_ms:
            return 0.0
        ordenadas = sorted(self.latencias_ms)
        meio = len(ordenadas) // 2
        if len(ordenadas) % 2:
            return ordenadas[meio]
        return (ordenadas[meio - 1] + ordenadas[meio]) / 2

    # -- serialização -------------------------------------------------------
    def para_dict(self, *, incluir_respostas: bool = False) -> dict[str, Any]:
        dados: dict[str, Any] = {
            "sistema": self.sistema,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "accuracy_entre_extraidos": round(self.accuracy_entre_extraidos, 4),
            "macro_f1": round(self.macro_f1, 4),
            "taxa_extracao": round(self.taxa_extracao, 4),
            "taxa_adesao_formato": round(self.taxa_adesao_formato, 4),
            "erros_de_execucao": self.erros,
            "latencia_media_ms": round(self.latencia_media_ms, 1),
            "latencia_mediana_ms": round(self.latencia_mediana_ms, 1),
            "por_classe": {
                r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                for r, d in self.f1_por_classe().items()
            },
            "matriz_confusao": self.matriz_confusao(),
            "distribuicao_previsoes": self.distribuicao_previsoes,
        }
        if incluir_respostas:
            dados["amostra_de_respostas"] = [
                {
                    "pergunta": p[:200],
                    "esperado": v,
                    "previsto": pr,
                    "resposta": r[:600],
                }
                for p, v, pr, r in list(
                    zip(self.perguntas, self.verdadeiros, self.previstos, self.respostas, strict=True)
                )[:20]
            ]
        return dados


def baseline_classe_majoritaria(verdadeiros: Sequence[str]) -> ResultadoAvaliacao:
    """
    Sistema de controle: responde sempre a classe mais frequente.

    POR QUE ISSO É INDISPENSÁVEL NO RELATÓRIO:
        É o piso contra o qual todo o resto deve ser lido. Um modelo com 58%
        de accuracy parece razoável até se descobrir que responder "yes" para
        tudo dá 55%. Sem esse número na tabela, o leitor não tem como julgar
        se o modelo aprendeu alguma coisa.
    """
    majoritaria = Counter(verdadeiros).most_common(1)[0][0]
    resultado = ResultadoAvaliacao(sistema=f"classe majoritária ('{majoritaria}')")
    resultado.verdadeiros = list(verdadeiros)
    resultado.previstos = [majoritaria] * len(verdadeiros)
    resultado.metodos = ["formato"] * len(verdadeiros)
    resultado.respostas = [f"Decisão: {majoritaria}"] * len(verdadeiros)
    resultado.perguntas = ["(baseline determinístico)"] * len(verdadeiros)
    return resultado


def tabela_comparativa(resultados: Sequence[ResultadoAvaliacao]) -> str:
    """Tabela em texto puro, pronta para o console e para o relatório."""
    cabecalho = (
        f"{'Sistema':<34} {'N':>5} {'Acc':>7} {'MacroF1':>8} "
        f"{'F1 yes':>7} {'F1 no':>7} {'F1 maybe':>9} {'Formato':>8} {'ms':>7}"
    )
    linhas = [cabecalho, "-" * len(cabecalho)]
    for r in resultados:
        f1 = r.f1_por_classe()
        linhas.append(
            f"{r.sistema:<34} {r.total:>5} {r.accuracy:>7.3f} {r.macro_f1:>8.3f} "
            f"{f1['yes']['f1']:>7.3f} {f1['no']['f1']:>7.3f} {f1['maybe']['f1']:>9.3f} "
            f"{r.taxa_adesao_formato:>7.1%} {r.latencia_media_ms:>7.0f}"
        )
    return "\n".join(linhas)
