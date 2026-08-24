#!/usr/bin/env python
"""
Etapa 5 — Construção do índice vetorial.

Fatia os protocolos internos e os abstracts do PubMedQA, calcula os embeddings
e grava o índice FAISS. Ao final, executa buscas de demonstração para mostrar
que a recuperação devolve trechos pertinentes com os marcadores de citação já
atribuídos.

Uso:
    make indexar
    python scripts/05_indexar.py --reconstruir
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.rag import indexar  # noqa: E402
from medgraph.rag.recuperador import Recuperador  # noqa: E402

console = Console()

# Perguntas de demonstração, escolhidas para exercitar as duas bases: as duas
# primeiras devem recuperar protocolo interno; a última, evidência científica.
PERGUNTAS_DEMO = [
    "Qual antibiótico empírico usar em sepse de foco urinário?",
    "Como ajustar a dose de enoxaparina em paciente com função renal alterada?",
    "Quais são os critérios para trombólise no AVC isquêmico agudo?",
    "Does aspirin reduce cardiovascular mortality?",
]


def main() -> int:
    reconstruir = "--reconstruir" in sys.argv or "--forcar" in sys.argv

    iniciar(
        banner="Etapa 5 — Índice vetorial",
        subtitulo="protocolos internos + evidência científica, com marcadores de citação",
    )
    cfg = obter_settings()

    console.print(Rule("[bold]Construção[/bold]"))
    estatisticas = indexar.construir(cfg, forcar=reconstruir)

    tabela = Table(show_header=True, header_style="bold cyan", title="Índice construído")
    tabela.add_column("Indicador", width=28)
    tabela.add_column("Valor", justify="right", width=34)
    tabela.add_row("Trechos indexados", f"{estatisticas.total_trechos:,}")
    for fonte, quantidade in (estatisticas.trechos_por_fonte or {}).items():
        tabela.add_row(f"  · {fonte}", f"{quantidade:,}")
    tabela.add_row("Documentos originais", f"{estatisticas.documentos_originais:,}")
    tabela.add_row("Caracteres totais", f"{estatisticas.caracteres_totais:,}")
    tabela.add_row("Modelo de embedding", estatisticas.modelo_embedding)
    tabela.add_row("Dimensão do vetor", str(estatisticas.dimensao))
    tabela.add_row("Tempo de construção", f"{estatisticas.duracao_s:.1f} s")
    console.print(tabela)

    console.print(Rule("[bold]Buscas de demonstração[/bold]"))
    recuperador = Recuperador(cfg)

    for pergunta in PERGUNTAS_DEMO:
        console.print(f"\n[bold cyan]?[/bold cyan] {pergunta}")
        trechos = recuperador.recuperar(pergunta)
        if not trechos:
            console.print("  [yellow]nenhum trecho acima do escore mínimo[/yellow]")
            continue
        for trecho in trechos:
            console.print(
                f"  [green][{trecho.marcador}][/green] "
                f"[dim]{trecho.escore:.3f}[/dim] · {trecho.rotulo_fonte}"
            )
            preview = " ".join(trecho.texto.split())[:150]
            console.print(f"      [dim]{preview}...[/dim]")

    console.print(
        "\n[bold green]Etapa 5 concluída.[/bold green] "
        "[dim]Cada trecho carrega o marcador que o modelo vai citar e o painel vai resolver.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
