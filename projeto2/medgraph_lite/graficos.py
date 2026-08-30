"""
Graficos da apresentacao.

Sao quatro, e cada um responde a uma pergunta que a banca faz:
  1. o treino funcionou?          -> curva de perda
  2. o ajuste melhorou o que?     -> antes x depois
  3. o fluxo faz o que promete?   -> caminho percorrido por consulta
  4. os limites funcionam?        -> achados por severidade
"""

from __future__ import annotations

import matplotlib.pyplot as plt

AZUL = "#1f4e79"
VERDE = "#2e7d32"
LARANJA = "#ef6c00"
VERMELHO = "#c62828"
CINZA = "#90a4ae"


def curva_de_perda(historico: list[dict], destino: str = "curva_de_perda.png"):
    """Perda caindo significa que o modelo esta aprendendo o formato."""
    passos = [(h["step"], h["loss"]) for h in historico if "loss" in h]
    if not passos:
        return None

    figura, eixo = plt.subplots(figsize=(8, 4))
    eixo.plot(*zip(*passos, strict=False), color=AZUL, linewidth=2, marker="o", markersize=4)
    eixo.set_xlabel("passo de treino")
    eixo.set_ylabel("perda")
    eixo.set_title("Fine-tuning QLoRA — perda de treino")
    eixo.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(destino, dpi=120)
    return figura


def antes_e_depois(resultados: dict[str, dict], destino: str = "antes_depois.png"):
    """
    Compara o modelo base com o ajustado nas duas metricas que importam.

    A adesao ao formato e o que o fine-tuning ensina; a acuracia mede se a
    decisao esta certa. Separa-las evita confundir "errou a resposta" com
    "nao seguiu o formato", que tem causas e correcoes diferentes.
    """
    sistemas = list(resultados)
    metricas = ["adesao_formato", "acuracia"]
    rotulos = ["Adesão ao formato", "Acurácia"]

    figura, eixos = plt.subplots(1, 2, figsize=(10, 4))
    for eixo, metrica, rotulo in zip(eixos, metricas, rotulos, strict=False):
        valores = [resultados[s][metrica] for s in sistemas]
        cores = [CINZA, VERDE][: len(sistemas)]
        barras = eixo.bar(sistemas, valores, color=cores, width=0.55)
        eixo.set_ylim(0, 1.05)
        eixo.set_title(rotulo)
        eixo.set_ylabel("proporção")
        eixo.grid(axis="y", alpha=0.3)
        for barra, valor in zip(barras, valores, strict=False):
            eixo.text(barra.get_x() + barra.get_width() / 2, valor + 0.03,
                      f"{valor:.0%}", ha="center", fontweight="bold")
    figura.suptitle("Modelo base × ajustado", fontweight="bold")
    figura.tight_layout()
    figura.savefig(destino, dpi=120)
    return figura


def caminho_do_grafo(trilhas: dict[str, list[dict]], destino: str = "caminhos.png"):
    """
    Mostra por quais nos cada consulta passou, e quanto tempo levou em cada um.

    E a evidencia visual de que o fluxo tem caminhos diferentes: uma pergunta
    recusada nao chega ao modelo, e uma com conflito passa pela validacao
    humana.
    """
    figura, eixo = plt.subplots(figsize=(11, 1.1 * len(trilhas) + 1.6))

    etapas_vistas: list[str] = []
    for trilha in trilhas.values():
        for evento in trilha:
            if evento["etapa"] not in etapas_vistas:
                etapas_vistas.append(evento["etapa"])

    for linha, trilha in enumerate(trilhas.values()):
        for evento in trilha:
            coluna = etapas_vistas.index(evento["etapa"])
            cor = VERMELHO if evento["etapa"] == "validacao_humana" else AZUL
            eixo.scatter(coluna, linha, s=380, color=cor, zorder=3)
            eixo.text(coluna, linha - 0.28, f"{evento['ms']:.0f}ms",
                      ha="center", fontsize=7, color="#555")
        colunas = [etapas_vistas.index(e["etapa"]) for e in trilha]
        eixo.plot(colunas, [linha] * len(colunas), color=CINZA, zorder=1, linewidth=1.5)

    eixo.set_yticks(range(len(trilhas)))
    eixo.set_yticklabels(list(trilhas), fontsize=9)
    # Primeiro caso em cima, na mesma ordem em que foram executados.
    eixo.invert_yaxis()
    # Margem embaixo para que o rotulo de latencia da ultima linha nao colida
    # com os nomes das etapas no eixo x.
    eixo.set_ylim(len(trilhas) - 0.45, -0.55)
    eixo.set_xticks(range(len(etapas_vistas)))
    eixo.set_xticklabels(etapas_vistas, rotation=30, ha="right", fontsize=8)
    eixo.set_title("Caminho de cada consulta no grafo", fontweight="bold")
    eixo.grid(axis="x", alpha=0.2)
    figura.tight_layout()
    figura.savefig(destino, dpi=120)
    return figura


def achados_por_severidade(contagem: dict[str, int], destino: str = "achados.png"):
    """Quantos alertas cada nivel de gravidade produziu nas consultas de teste."""
    ordem = ["critico", "atencao", "informativo"]
    cores = {"critico": VERMELHO, "atencao": LARANJA, "informativo": CINZA}
    presentes = [s for s in ordem if contagem.get(s)]

    figura, eixo = plt.subplots(figsize=(6, 3.6))
    barras = eixo.barh(presentes, [contagem[s] for s in presentes],
                       color=[cores[s] for s in presentes], height=0.55)
    for barra, sev in zip(barras, presentes, strict=False):
        eixo.text(barra.get_width() + 0.08, barra.get_y() + barra.get_height() / 2,
                  str(contagem[sev]), va="center", fontweight="bold")
    eixo.invert_yaxis()
    eixo.set_xlabel("ocorrências")
    eixo.set_title("Achados dos guardrails, por severidade", fontweight="bold")
    eixo.grid(axis="x", alpha=0.3)
    figura.tight_layout()
    figura.savefig(destino, dpi=120)
    return figura
