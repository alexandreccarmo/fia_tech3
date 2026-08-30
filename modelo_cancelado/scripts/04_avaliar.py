#!/usr/bin/env python
"""
Etapa 4 — Avaliação do modelo e análise dos resultados.

O QUE ESTE SCRIPT FAZ:
    Roda o conjunto de teste isolado na Etapa 1 através de até quatro
    sistemas, calcula as métricas, gera os gráficos do relatório e apresenta a
    tabela comparativa.

O CONJUNTO DE TESTE:
    500 exemplos do PubMedQA anotados por especialistas, separados na curadoria
    e nunca utilizados no treino. Há uma verificação automática de vazamento em
    `preparar_dataset_sft.py` que falharia se qualquer um deles tivesse
    escapado para o dataset de fine-tuning.

Uso:
    make avaliar
    python scripts/04_avaliar.py                    # 150 casos (~15 min)
    python scripts/04_avaliar.py --completo         # os 500 casos (~1h/sistema)
    python scripts/04_avaliar.py --rapido           # 30 casos, para conferir o fluxo
    python scripts/04_avaliar.py --sem-openai       # não gasta nada
"""

from __future__ import annotations

import argparse
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
from medgraph.auditoria import abrir_trilha  # noqa: E402
from medgraph.avaliacao import avaliar, graficos  # noqa: E402
from medgraph.avaliacao.metricas import ROTULOS  # noqa: E402
from medgraph.llm.custo import contador  # noqa: E402

console = Console()


def _tabela_resultados(resultados) -> Table:
    tabela = Table(show_header=True, header_style="bold cyan", title="Comparativo dos sistemas")
    tabela.add_column("Sistema", width=30)
    tabela.add_column("N", justify="right", width=5)
    tabela.add_column("Accuracy", justify="right", width=9)
    tabela.add_column("Macro-F1", justify="right", width=9)
    tabela.add_column("F1 yes", justify="right", width=7)
    tabela.add_column("F1 no", justify="right", width=7)
    tabela.add_column("F1 maybe", justify="right", width=9)
    tabela.add_column("Formato", justify="right", width=8)
    tabela.add_column("ms", justify="right", width=7)

    melhor_macro = max((r.macro_f1 for r in resultados), default=0)
    for r in resultados:
        f1 = r.f1_por_classe()
        destaque = "[bold green]" if r.macro_f1 == melhor_macro and melhor_macro > 0 else ""
        fecha = "[/bold green]" if destaque else ""
        tabela.add_row(
            r.sistema,
            str(r.total),
            f"{r.accuracy:.3f}",
            f"{destaque}{r.macro_f1:.3f}{fecha}",
            f"{f1['yes']['f1']:.3f}",
            f"{f1['no']['f1']:.3f}",
            f"{f1['maybe']['f1']:.3f}",
            f"{r.taxa_adesao_formato:.0%}",
            f"{r.latencia_media_ms:.0f}",
        )
    return tabela


def _tabela_distribuicao(resultados) -> Table:
    """
    Como cada sistema distribuiu suas respostas.

    Delata na hora o modelo que colapsou numa classe só — o modo de falha mais
    comum e o mais fácil de confundir com bom desempenho.
    """
    tabela = Table(
        show_header=True, header_style="bold cyan",
        title="Distribuição das previsões (o modelo usa as três classes?)",
    )
    tabela.add_column("Sistema", width=30)
    for rotulo in ROTULOS:
        tabela.add_column(rotulo, justify="right", width=8)
    tabela.add_column("sem rótulo", justify="right", width=11)

    for r in resultados:
        distribuicao = r.distribuicao_previsoes
        tabela.add_row(
            r.sistema,
            *[str(distribuicao.get(rotulo, 0)) for rotulo in ROTULOS],
            str(distribuicao.get("(sem rótulo)", 0)),
        )
    return tabela


def _analise(resultados) -> list[str]:
    """
    Leitura dos números — o que a tabela sozinha não diz.

    Gerada a partir dos resultados, e não escrita à mão, para que continue
    verdadeira quando o modelo ajustado entrar na comparação.
    """
    linhas: list[str] = []

    piso = next((r for r in resultados if "majorit" in r.sistema.lower()), None)
    base = next((r for r in resultados if r.sistema.startswith("modelo base")), None)
    ajustado = next((r for r in resultados if r.sistema.startswith("modelo ajustado")), None)
    teto = next((r for r in resultados if "referência" in r.sistema), None)

    if piso:
        linhas.append(
            f"O piso é {piso.accuracy:.1%} de accuracy — o que se obtém respondendo "
            f"sempre a classe mais frequente. O macro-F1 desse mesmo sistema é "
            f"{piso.macro_f1:.3f}, e a distância entre os dois números mostra por que "
            f"a accuracy sozinha não serve para julgar este problema."
        )

    if base and piso:
        delta = base.accuracy - piso.accuracy
        veredito = "acima" if delta > 0.02 else ("na prática igual" if abs(delta) <= 0.02 else "abaixo")
        linhas.append(
            f"O modelo base ficou {veredito} do piso ({base.accuracy:.1%} contra "
            f"{piso.accuracy:.1%}), com macro-F1 de {base.macro_f1:.3f} e aderência ao "
            f"formato de {base.taxa_adesao_formato:.0%}."
        )

    if ajustado and base:
        d_acc = ajustado.accuracy - base.accuracy
        d_f1 = ajustado.macro_f1 - base.macro_f1
        d_formato = ajustado.taxa_adesao_formato - base.taxa_adesao_formato
        linhas.append(
            f"O fine-tuning moveu a accuracy em {d_acc:+.1%} e o macro-F1 em {d_f1:+.3f}. "
            f"A aderência ao formato variou {d_formato:+.0%}, que é o efeito mais direto "
            f"do treino sobre o comportamento do modelo."
        )
        f1_maybe_base = base.f1_por_classe()["maybe"]["f1"]
        f1_maybe_ajustado = ajustado.f1_por_classe()["maybe"]["f1"]
        if f1_maybe_base < 0.05 <= f1_maybe_ajustado:
            linhas.append(
                f"A classe 'maybe' saiu de F1 {f1_maybe_base:.3f} para {f1_maybe_ajustado:.3f}: "
                f"o modelo passou a reconhecer evidência inconclusiva em vez de forçar uma "
                f"resposta definitiva. É o ganho que a repetição por classe no dataset visava."
            )
    elif base and not ajustado:
        linhas.append(
            "O modelo ajustado ainda não foi avaliado: o fine-tuning é executado no "
            "Google Colab. Assim que o GGUF for registrado no Ollama, esta mesma tabela "
            "passa a trazer as duas colunas e a comparação fica completa."
        )

    if teto:
        linhas.append(
            f"O teto de referência ({teto.sistema}) alcançou {teto.accuracy:.1%} de accuracy "
            f"e macro-F1 de {teto.macro_f1:.3f}, sobre {teto.total} casos. Serve para dar "
            f"escala: é um modelo cerca de cem vezes maior, executado em nuvem paga."
        )

    linhas.append(
        "Referência externa: especialistas humanos alcançam 78% de acurácia neste mesmo "
        "conjunto, segundo o artigo original do PubMedQA (Jin et al., 2019)."
    )

    for r in resultados:
        if r.erros:
            linhas.append(
                f"Atenção: {r.erros} caso(s) de '{r.sistema}' falharam por erro de execução "
                f"e foram contabilizados como não respondidos."
            )
    return linhas


def main() -> int:
    analisador = argparse.ArgumentParser(description="Avalia os sistemas no PubMedQA.")
    analisador.add_argument("--completo", action="store_true", help="usa os 500 casos de teste")
    analisador.add_argument("--rapido", action="store_true", help="usa 30 casos, só para conferir o fluxo")
    analisador.add_argument("--sem-openai", action="store_true", help="não chama a API paga")
    analisador.add_argument("--n", type=int, default=None, help="número de casos")
    argumentos = analisador.parse_args()

    if argumentos.rapido:
        n_local, n_openai = 30, 20
    elif argumentos.completo:
        n_local, n_openai = 500, 100
    else:
        n_local = argumentos.n or avaliar.AMOSTRA_PADRAO_LOCAL
        n_openai = min(n_local, avaliar.AMOSTRA_PADRAO_OPENAI)

    iniciar(
        banner="Etapa 4 — Avaliação do modelo",
        subtitulo=f"{n_local} casos de teste jamais vistos no treino",
    )
    cfg = obter_settings()

    with abrir_trilha(pergunta="[pipeline] avaliação", usuario="sistema") as trilha:
        saida = avaliar.executar(
            cfg,
            n_local=n_local,
            n_openai=n_openai,
            incluir_openai=not argumentos.sem_openai,
        )
        trace_id = trilha.trace_id

    resultados = saida["resultados"]

    console.print(Rule("[bold green]Resultados[/bold green]", style="green"))
    console.print(_tabela_resultados(resultados))
    console.print()
    console.print(_tabela_distribuicao(resultados))

    console.print(Rule("[bold]Análise[/bold]"))
    for i, linha in enumerate(_analise(resultados), 1):
        console.print(f"  [cyan]{i}.[/cyan] {linha}\n")

    console.print(Rule("[bold]Gráficos[/bold]"))
    for caminho in graficos.gerar_todos(resultados, cfg.dir_graficos):
        console.print(f"  {Path(caminho).relative_to(cfg.dir_raiz)}")

    console.print(Rule("[bold]Custo desta execução[/bold]"))
    console.print(contador(cfg).tabela_resumo())

    console.print(
        f"\n[dim]resultados completos: docs/avaliacao_resultados.json[/dim]"
        f"\n[dim]trace da execução: logs/traces/{trace_id}.json[/dim]"
    )
    console.print("\n[bold green]Etapa 4 concluída.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
