"""
Graficos da apresentacao.

Sao quatro, e cada um responde a uma pergunta que a banca faz:
  1. o treino funcionou?          -> curva de perda
  2. o ajuste melhorou o que?     -> antes x depois
  3. o fluxo faz o que promete?   -> caminho percorrido por consulta
  4. onde esta o custo?           -> linha do tempo por consulta
  5. os limites funcionam?        -> achados por severidade
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


# =============================================================================
# O GRAFO COM O CAMINHO PERCORRIDO
# =============================================================================
# Posicoes fixas dos nos. O grafo tem nove nos e nao muda, entao um layout
# escrito a mao comunica melhor do que qualquer algoritmo automatico: a coluna
# central e o caminho feliz, e os desvios saem para os lados.
#
# O teste `test_desenho_cobre_todos_os_nos_do_grafo` compara este dicionario com
# os nos do grafo compilado: um no acrescentado la e esquecido aqui sairia
# apagado na figura, como se nao tivesse executado.
POSICOES = {
    "guardrail_entrada":    (0.0, 12.0),
    "consultar_prontuario": (0.0, 10.5),
    "verificar_exames":     (0.0, 9.0),
    "recuperar_evidencia":  (0.0, 7.5),
    "responder":            (0.0, 6.0),
    "verificar_resposta":   (0.0, 4.5),
    "emitir_alerta":        (1.15, 3.0),
    "validacao_humana":     (1.15, 1.5),
    "montar_resposta":      (0.0, 0.0),
}

# (origem, destino, curvatura). A curvatura afasta a aresta da linha reta, para
# que o desvio da recusa nao passe por cima dos nos do meio.
ARESTAS = [
    ("guardrail_entrada", "consultar_prontuario", 0.0),
    ("guardrail_entrada", "montar_resposta", -0.62),
    ("consultar_prontuario", "verificar_exames", 0.0),
    ("verificar_exames", "recuperar_evidencia", 0.0),
    ("recuperar_evidencia", "responder", 0.0),
    ("responder", "verificar_resposta", 0.0),
    ("verificar_resposta", "montar_resposta", 0.0),
    ("verificar_resposta", "emitir_alerta", 0.0),
    ("emitir_alerta", "validacao_humana", 0.0),
    ("validacao_humana", "montar_resposta", 0.0),
]

ROTULOS = {
    "guardrail_entrada": "guardrail\nentrada",
    "consultar_prontuario": "consultar\nprontuário",
    "verificar_exames": "verificar\nexames",
    "recuperar_evidencia": "recuperar\nevidência",
    "responder": "responder\n(LLM)",
    "verificar_resposta": "verificar\nresposta",
    "emitir_alerta": "emitir\nalerta",
    "validacao_humana": "validação\nhumana",
    "montar_resposta": "montar\nresposta",
}

# Nos que sinalizam risco. Saem em vermelho quando visitados, para que o desvio
# se leia na figura sem precisar do rotulo.
NOS_DE_RISCO = {"emitir_alerta", "validacao_humana"}


def _desenhar_aresta(eixo, origem, destino, curvatura, cor, largura, alfa):
    from matplotlib.patches import FancyArrowPatch

    eixo.add_patch(FancyArrowPatch(
        POSICOES[origem], POSICOES[destino],
        connectionstyle=f"arc3,rad={curvatura}",
        arrowstyle="-|>", mutation_scale=13,
        color=cor, linewidth=largura, alpha=alfa,
        shrinkA=26, shrinkB=26, zorder=1,
    ))


def fluxo_percorrido(trilhas: dict[str, list[dict]], destino: str = "fluxo.png"):
    """
    Desenha o grafo uma vez por consulta, destacando o caminho que ela seguiu.

    E a figura que responde "o fluxo decide alguma coisa?" sem precisar de
    explicacao: o que ficou apagado nao foi executado. Ver a consulta recusada
    saltar direto do guardrail para a resposta final, sem passar pela LLM, diz
    mais do que qualquer descricao do roteamento.
    """
    figura, eixos = plt.subplots(1, len(trilhas), figsize=(3.5 * len(trilhas), 9.4))
    if len(trilhas) == 1:
        eixos = [eixos]

    for eixo, (nome, trilha) in zip(eixos, trilhas.items(), strict=False):
        visitados = [evento["etapa"] for evento in trilha]
        tempos = {e["etapa"]: e["ms"] for e in trilha}
        percorridas = set(zip(visitados, visitados[1:], strict=False))

        for origem, alvo, curvatura in ARESTAS:
            ativa = (origem, alvo) in percorridas
            _desenhar_aresta(
                eixo, origem, alvo, curvatura,
                cor=AZUL if ativa else "#dde3e8",
                largura=2.4 if ativa else 1.0,
                alfa=1.0 if ativa else 0.9,
            )

        for etapa, (x, y) in POSICOES.items():
            visitado = etapa in visitados
            if not visitado:
                cor, texto, borda = "#f2f5f7", "#b0bec5", "#dde3e8"
            elif etapa in NOS_DE_RISCO:
                cor, texto, borda = VERMELHO, "white", VERMELHO
            else:
                cor, texto, borda = AZUL, "white", AZUL

            eixo.scatter(x, y, s=2600, color=cor, edgecolors=borda,
                         linewidths=1.6, zorder=2)
            eixo.text(x, y, ROTULOS[etapa], ha="center", va="center",
                      fontsize=7.2, color=texto, fontweight="bold", zorder=3)
            if visitado:
                eixo.text(x + 0.42, y, f"{tempos[etapa]:.0f} ms", ha="left",
                          va="center", fontsize=6.4, color="#607d8b", zorder=3)

        parou = "validacao_humana" in visitados
        eixo.set_title(
            f"{nome}\n{len(visitados)} nós" + ("  ·  RETIDA" if parou else ""),
            fontsize=9.5, fontweight="bold",
            color=VERMELHO if parou else "#37474f", pad=12,
        )
        eixo.set_xlim(-0.85, 2.55)
        eixo.set_ylim(-1.0, 13.0)
        eixo.axis("off")

    figura.suptitle("Caminho percorrido no grafo, por consulta",
                    fontsize=13, fontweight="bold")
    figura.tight_layout()
    figura.savefig(destino, dpi=130, bbox_inches="tight")
    return figura
