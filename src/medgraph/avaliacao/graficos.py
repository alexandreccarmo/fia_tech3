"""
[REQ-E3] Gráficos da avaliação.

O QUE FAZ:
    Gera as figuras que entram no relatório técnico e no painel visual.

POR QUE GRÁFICO E NÃO SÓ TABELA:
    A tabela tem os números; o gráfico tem o argumento. A comparação entre
    accuracy e macro-F1, lado a lado, mostra em um olhar o que a tabela exige
    ler duas colunas para perceber: que um sistema pode ter accuracy alta e
    ainda assim ser inútil, porque colapsou numa única classe.

DECISÕES DE APRESENTAÇÃO:
    - Accuracy e macro-F1 sempre juntos, nunca isolados.
    - A linha do piso (classe majoritária) atravessa o gráfico de barras: é a
      referência contra a qual todo o resto deve ser lido.
    - A linha do especialista humano (78%) dá o horizonte da tarefa.
    - A matriz de confusão é normalizada por linha, para que a classe "maybe",
      com 11% dos casos, seja legível ao lado de "yes", com 55%.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem interface gráfica: os scripts rodam em terminal
import matplotlib.pyplot as plt  # noqa: E402

from medgraph.avaliacao.metricas import ROTULOS, ResultadoAvaliacao  # noqa: E402
from medgraph.logging_config import obter_logger  # noqa: E402

log = obter_logger(__name__)

# Paleta consistente entre todas as figuras do relatório.
COR_ACCURACY = "#4C72B0"
COR_MACRO_F1 = "#DD8452"
COR_PISO = "#C44E52"
COR_HUMANO = "#55A868"
REFERENCIA_HUMANO = 0.78


def _abreviar(nome: str, limite: int = 26) -> str:
    return nome if len(nome) <= limite else nome[: limite - 1] + "…"


def comparativo_de_sistemas(
    resultados: Sequence[ResultadoAvaliacao], destino: str | Path
) -> str:
    """Barras de accuracy e macro-F1 por sistema, com as duas linhas de referência."""
    nomes = [_abreviar(r.sistema) for r in resultados]
    accuracies = [r.accuracy for r in resultados]
    macros = [r.macro_f1 for r in resultados]

    posicoes = range(len(nomes))
    largura = 0.38

    figura, eixo = plt.subplots(figsize=(max(9, len(nomes) * 2.4), 5.5))
    eixo.bar(
        [p - largura / 2 for p in posicoes], accuracies, largura,
        label="accuracy", color=COR_ACCURACY,
    )
    eixo.bar(
        [p + largura / 2 for p in posicoes], macros, largura,
        label="macro-F1", color=COR_MACRO_F1,
    )

    # O piso: a accuracy da classe majoritária. Tudo abaixo disso é ruído.
    piso = next((r.accuracy for r in resultados if "majorit" in r.sistema.lower()), None)
    if piso is not None:
        eixo.axhline(
            piso, color=COR_PISO, linestyle="--", linewidth=1.4,
            label=f"piso — classe majoritária ({piso:.1%})",
        )

    eixo.axhline(
        REFERENCIA_HUMANO, color=COR_HUMANO, linestyle=":", linewidth=1.6,
        label=f"especialista humano ({REFERENCIA_HUMANO:.0%})",
    )

    for posicao, (a, m) in enumerate(zip(accuracies, macros, strict=True)):
        eixo.text(posicao - largura / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
        eixo.text(posicao + largura / 2, m + 0.015, f"{m:.3f}", ha="center", fontsize=9)

    eixo.set_xticks(list(posicoes))
    eixo.set_xticklabels(nomes, rotation=12, ha="right", fontsize=9)
    eixo.set_ylabel("desempenho")
    eixo.set_ylim(0, 1.0)
    eixo.set_title(
        "Desempenho no PubMedQA — conjunto de teste anotado por especialistas",
        fontsize=12,
    )
    eixo.legend(loc="upper left", fontsize=9)
    eixo.grid(axis="y", alpha=0.3)
    figura.tight_layout()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return str(destino)


def matriz_de_confusao(resultado: ResultadoAvaliacao, destino: str | Path) -> str:
    """
    Matriz verdadeiro × previsto, normalizada por linha.

    A normalização é o que torna a figura legível: sem ela, a linha de "maybe"
    (55 casos) desapareceria ao lado da de "yes" (276 casos), e justamente a
    classe mais difícil ficaria invisível.
    """
    matriz = resultado.matriz_confusao()
    colunas = [*ROTULOS, "(sem rótulo)"]

    dados: list[list[float]] = []
    for verdadeiro in ROTULOS:
        total = sum(matriz[verdadeiro].values()) or 1
        dados.append([matriz[verdadeiro][c] / total for c in colunas])

    figura, eixo = plt.subplots(figsize=(7.5, 5.5))
    imagem = eixo.imshow(dados, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    eixo.set_xticks(range(len(colunas)))
    eixo.set_xticklabels(colunas)
    eixo.set_yticks(range(len(ROTULOS)))
    eixo.set_yticklabels(
        [f"{r}\n(n={sum(matriz[r].values())})" for r in ROTULOS]
    )
    eixo.set_xlabel("previsto pelo modelo")
    eixo.set_ylabel("rótulo do especialista")
    eixo.set_title(f"Matriz de confusão — {resultado.sistema}", fontsize=11)

    for i, verdadeiro in enumerate(ROTULOS):
        for j, coluna in enumerate(colunas):
            quantidade = matriz[verdadeiro][coluna]
            proporcao = dados[i][j]
            eixo.text(
                j, i,
                f"{quantidade}\n{proporcao:.0%}",
                ha="center", va="center",
                color="white" if proporcao > 0.5 else "black",
                fontsize=10,
            )

    figura.colorbar(imagem, ax=eixo, label="proporção da linha")
    figura.tight_layout()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return str(destino)


def f1_por_classe(resultados: Sequence[ResultadoAvaliacao], destino: str | Path) -> str:
    """
    F1 de cada classe, por sistema.

    É a figura que mostra ONDE o ganho aconteceu. Um aumento de macro-F1 pode
    vir de melhora uniforme ou de o modelo finalmente ter passado a prever
    "maybe" — e essas duas histórias são completamente diferentes.
    """
    nomes = [_abreviar(r.sistema, 22) for r in resultados]
    posicoes = range(len(nomes))
    largura = 0.26
    cores = {"yes": COR_ACCURACY, "no": COR_MACRO_F1, "maybe": COR_HUMANO}

    figura, eixo = plt.subplots(figsize=(max(9, len(nomes) * 2.4), 5))
    for deslocamento, rotulo in zip((-largura, 0, largura), ROTULOS, strict=True):
        valores = [r.f1_por_classe()[rotulo]["f1"] for r in resultados]
        eixo.bar(
            [p + deslocamento for p in posicoes], valores, largura,
            label=f"F1 — {rotulo}", color=cores[rotulo],
        )
        for posicao, valor in zip(posicoes, valores, strict=True):
            if valor > 0.02:
                eixo.text(
                    posicao + deslocamento, valor + 0.015, f"{valor:.2f}",
                    ha="center", fontsize=8,
                )

    eixo.set_xticks(list(posicoes))
    eixo.set_xticklabels(nomes, rotation=12, ha="right", fontsize=9)
    eixo.set_ylabel("F1")
    eixo.set_ylim(0, 1.0)
    eixo.set_title("F1 por classe — onde está o ganho", fontsize=12)
    eixo.legend(fontsize=9)
    eixo.grid(axis="y", alpha=0.3)
    figura.tight_layout()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return str(destino)


def adesao_e_latencia(resultados: Sequence[ResultadoAvaliacao], destino: str | Path) -> str:
    """
    Aderência ao formato e latência média, lado a lado.

    A aderência ao formato é evidência direta do efeito do fine-tuning sobre o
    COMPORTAMENTO do modelo, independentemente de ele acertar a resposta. A
    latência é o custo operacional de cada opção — e é o que justifica servir
    um modelo de 3B localmente em vez de chamar uma API.
    """
    nomes = [_abreviar(r.sistema, 22) for r in resultados]
    posicoes = range(len(nomes))

    figura, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.8))

    adesoes = [r.taxa_adesao_formato for r in resultados]
    esquerda.bar(posicoes, adesoes, 0.55, color=COR_ACCURACY)
    for posicao, valor in zip(posicoes, adesoes, strict=True):
        esquerda.text(posicao, valor + 0.02, f"{valor:.0%}", ha="center", fontsize=9)
    esquerda.set_xticks(list(posicoes))
    esquerda.set_xticklabels(nomes, rotation=18, ha="right", fontsize=8)
    esquerda.set_ylabel("respostas com a linha 'Decisão:'")
    esquerda.set_ylim(0, 1.12)
    esquerda.set_title("Aderência ao formato de saída", fontsize=11)
    esquerda.grid(axis="y", alpha=0.3)

    latencias = [r.latencia_media_ms for r in resultados]
    direita.bar(posicoes, latencias, 0.55, color=COR_MACRO_F1)
    for posicao, valor in zip(posicoes, latencias, strict=True):
        direita.text(posicao, valor * 1.02, f"{valor:.0f}", ha="center", fontsize=9)
    direita.set_xticks(list(posicoes))
    direita.set_xticklabels(nomes, rotation=18, ha="right", fontsize=8)
    direita.set_ylabel("milissegundos por consulta")
    direita.set_title("Latência média", fontsize=11)
    direita.grid(axis="y", alpha=0.3)

    figura.tight_layout()
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return str(destino)


def gerar_todos(resultados: Sequence[ResultadoAvaliacao], pasta: str | Path) -> list[str]:
    """Gera o conjunto completo de figuras da avaliação."""
    pasta = Path(pasta)
    gerados = [
        comparativo_de_sistemas(resultados, pasta / "comparativo_sistemas.png"),
        f1_por_classe(resultados, pasta / "f1_por_classe.png"),
        adesao_e_latencia(resultados, pasta / "adesao_e_latencia.png"),
    ]

    # Uma matriz por sistema que efetivamente chamou um modelo. A matriz do
    # baseline seria uma coluna única e não acrescenta nada.
    for resultado in resultados:
        if "majorit" in resultado.sistema.lower():
            continue
        arquivo = "matriz_" + resultado.sistema.split()[0].lower().replace("-", "_") + ".png"
        gerados.append(matriz_de_confusao(resultado, pasta / arquivo))

    for caminho in gerados:
        log.info("gráfico gerado: %s", caminho)
    return gerados
