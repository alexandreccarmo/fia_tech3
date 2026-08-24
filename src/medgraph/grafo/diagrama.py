"""
[REQ-E3] Geração dos diagramas do fluxo.

O QUE FAZ:
    Produz três representações do mesmo grafo, em `docs/diagramas/`:

      grafo_ascii.txt   desenho em texto, via `draw_ascii()` do LangGraph
      grafo.mmd         código Mermaid, via `draw_mermaid()`
      grafo.png         imagem, renderizada a partir do Mermaid

POR QUE TRÊS FORMATOS:
    Cada um serve a um consumidor diferente. O ASCII vai para o terminal
    durante a demonstração e para o README — é o mesmo `draw_ascii()`
    apresentado nas aulas, e não depende de nada externo. O Mermaid é texto
    versionável: uma mudança no fluxo aparece como diff legível no Git. O PNG
    entra no relatório técnico e nos slides.

O PNG DEPENDE DE REDE — E ISSO É TRATADO:
    `draw_mermaid_png()` do LangGraph renderiza através do serviço mermaid.ink.
    Sem internet, a chamada falha. O fluxo aqui trata isso como situação
    esperada e não como erro: o `.mmd` continua sendo gravado, e a mensagem
    diz como renderizá-lo depois. Um gerador de documentação que quebra o
    build por falta de rede seria pior do que a ausência da imagem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.grafo.construir import compilar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


def gerar(cfg: Settings | None = None) -> dict[str, Any]:
    """Gera os três arquivos e devolve o que foi produzido."""
    cfg = cfg or obter_settings()
    pasta = cfg.dir_diagramas
    pasta.mkdir(parents=True, exist_ok=True)

    # Sem checkpointer: gerar diagrama não deve criar banco de dados nem
    # depender de estado persistido.
    app = compilar(cfg, com_checkpointer=False, com_validacao_humana=False)
    grafo = app.get_graph()

    resultado: dict[str, Any] = {"arquivos": [], "avisos": []}

    # --- ASCII --------------------------------------------------------------
    ascii_arte = grafo.draw_ascii()
    destino_ascii = pasta / "grafo_ascii.txt"
    destino_ascii.write_text(
        "MedGraph — fluxo de decisão clínica (LangGraph)\n"
        "Gerado por: make diagrama\n"
        + "=" * 78 + "\n\n"
        + ascii_arte,
        encoding="utf-8",
    )
    resultado["arquivos"].append(str(destino_ascii))
    resultado["ascii"] = ascii_arte
    log.info("ASCII gravado em %s", destino_ascii)

    # --- Mermaid ------------------------------------------------------------
    mermaid = grafo.draw_mermaid()
    destino_mmd = pasta / "grafo.mmd"
    destino_mmd.write_text(mermaid, encoding="utf-8")
    resultado["arquivos"].append(str(destino_mmd))
    resultado["mermaid"] = mermaid
    log.info("Mermaid gravado em %s", destino_mmd)

    # --- PNG ----------------------------------------------------------------
    destino_png = pasta / "grafo.png"
    try:
        png = grafo.draw_mermaid_png()
        destino_png.write_bytes(png)
        resultado["arquivos"].append(str(destino_png))
        log.info("PNG gravado em %s (%.0f KB)", destino_png, len(png) / 1024)
    except Exception as exc:
        aviso = (
            f"PNG não gerado ({type(exc).__name__}). A renderização usa o serviço "
            f"mermaid.ink e exige internet. O arquivo grafo.mmd foi gravado e pode "
            f"ser renderizado em https://mermaid.live ou por qualquer editor com "
            f"suporte a Mermaid."
        )
        resultado["avisos"].append(aviso)
        log.warning(aviso)

    return resultado


def imprimir_ascii(cfg: Settings | None = None) -> str:
    """Desenha o grafo no terminal. Usado na abertura da demonstração."""
    app = compilar(cfg, com_checkpointer=False, com_validacao_humana=False)
    ascii_arte = app.get_graph().draw_ascii()
    print(ascii_arte)
    return ascii_arte


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Diagramas do grafo", subtitulo="ASCII, Mermaid e PNG")
    saida = gerar()
    for arquivo in saida["arquivos"]:
        print("gerado:", Path(arquivo).name)
    for aviso in saida["avisos"]:
        print("aviso :", aviso)
