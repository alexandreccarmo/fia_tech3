#!/usr/bin/env python
"""
Etapa 2 — Preparação do fine-tuning.

O QUE ESTE SCRIPT FAZ:
    Remonta o dataset supervisionado a partir dos dados curados, apresenta a
    composição resultante e imprime o passo a passo do treino no Google Colab —
    que é a única etapa do projeto que não roda nesta máquina.

POR QUE O TREINO NÃO RODA AQUI:
    Ajustar um modelo de 3 bilhões de parâmetros exige uma GPU com CUDA. Na T4
    gratuita do Colab o treino leva cerca de uma hora; no Apple Silicon levaria
    várias, e sem `bitsandbytes` não haveria quantização em 4 bits — o modelo
    simplesmente não caberia na memória.

    O que sai de lá é um adapter de ~50 MB, que volta para cá, é fundido ao
    modelo base, convertido para GGUF e servido pelo Ollama.

QUANDO REEXECUTAR:
    Sempre que a curadoria mudar, o corpus sintético crescer, ou o prompt de
    sistema for alterado. Esta última é a mais fácil de esquecer e a mais
    custosa: o prompt está gravado dentro de cada exemplo de treino, e um
    dataset montado com um prompt e um modelo consultado com outro produz
    exatamente o sintoma que este projeto quer evitar — o abandono silencioso
    do formato de citação.

Uso:
    make finetune-prep
    python scripts/02_preparar_finetune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.auditoria import abrir_trilha  # noqa: E402
from medgraph.chains import prompts  # noqa: E402
from medgraph.finetune import preparar_dataset_sft  # noqa: E402

console = Console()


def _tabela_composicao(resumo) -> Table:
    tabela = Table(show_header=True, header_style="bold cyan", title="Composição do dataset")
    tabela.add_column("Origem", width=26)
    tabela.add_column("Exemplos", justify="right", width=10)
    tabela.add_column("O que ensina", overflow="fold")

    papeis = {
        "pubmedqa_artificial": "o formato da tarefa — ler evidência, decidir, citar",
        "pubmedqa_especialista": "qualidade e a classe 'maybe', ausente do artificial",
        "faq_medicos": "o domínio e o idioma português",
        "modelos_documentos": "o formato institucional e o limite da prescrição",
    }
    for origem, quantidade in sorted(resumo["por_origem"].items(), key=lambda x: -x[1]):
        tabela.add_row(origem, f"{quantidade:,}", papeis.get(origem, ""))
    tabela.add_row("[bold]TOTAL[/bold]", f"[bold]{resumo['total']:,}[/bold]", "")
    return tabela


def _tabela_rotulos(resumo) -> Table:
    tabela = Table(show_header=True, header_style="bold cyan", title="Distribuição dos rótulos")
    tabela.add_column("Rótulo", width=14)
    tabela.add_column("Exemplos", justify="right", width=10)
    tabela.add_column("Proporção", justify="right", width=11)
    tabela.add_column("Repetição aplicada", justify="right", width=18)

    repeticoes = resumo.get("repeticoes_aplicadas", {})
    for rotulo, quantidade in sorted(resumo["por_rotulo"].items(), key=lambda x: -x[1]):
        fator = repeticoes.get(rotulo)
        tabela.add_row(
            rotulo,
            f"{quantidade:,}",
            f"{resumo['proporcao_por_rotulo'][rotulo]:.1%}",
            f"×{fator}" if fator else "—",
        )
    return tabela


def main() -> int:
    iniciar(
        banner="Etapa 2 — Preparação do fine-tuning",
        subtitulo="dataset supervisionado no formato de conversa, para o Colab",
    )
    cfg = obter_settings()

    with abrir_trilha(pergunta="[pipeline] preparação do fine-tuning", usuario="sistema"):
        resumo = preparar_dataset_sft.executar(cfg)

    console.print(Rule("[bold green]Dataset montado[/bold green]", style="green"))
    console.print(_tabela_composicao(resumo))
    console.print()
    console.print(_tabela_rotulos(resumo))

    console.print(
        f"\n  treino ........... {resumo['treino']:,} exemplos"
        f"\n  validação ........ {resumo['validacao']:,} exemplos"
        f"\n  tamanho médio .... {resumo['caracteres_por_exemplo']['media']:,} caracteres"
        f"\n  tokens estimados . ~{resumo['tokens_estimados_total'] / 1e6:.1f} M por época"
        f"\n  tempo estimado ... ~{resumo['tokens_estimados_total'] / 1e6 * 12:.0f} min por época na T4"
    )

    # -------------------------------------------------------------------------
    # Verificação de consistência do prompt
    # -------------------------------------------------------------------------
    # O prompt de sistema fica GRAVADO dentro de cada exemplo. Se ele mudou
    # desde a última montagem e o dataset não foi refeito, o modelo será
    # treinado sob um contrato e consultado sob outro.
    # -------------------------------------------------------------------------
    import json

    caminho_treino = cfg.dir_dados_processados / "sft_train.jsonl"
    with caminho_treino.open(encoding="utf-8") as arquivo:
        primeiro = json.loads(next(arquivo))
    sistema_no_dataset = primeiro["messages"][0]["content"]

    if sistema_no_dataset.strip() == prompts.SISTEMA.strip():
        console.print(
            "\n[green]✓[/green] O prompt de sistema gravado no dataset é idêntico ao "
            "de `medgraph.chains.prompts`."
        )
    else:
        console.print(
            "\n[red]✗ DIVERGÊNCIA:[/red] o prompt gravado no dataset difere do atual. "
            "Isso treinaria o modelo sob um contrato e o consultaria sob outro."
        )
        return 1

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]Próximo passo — no Google Colab[/bold]"))
    console.print(
        Panel(
            "[bold]1.[/bold] Aceite a licença do Llama 3.2 (gratuito, aprovação imediata)\n"
            "   [cyan]https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct[/cyan]\n\n"
            "[bold]2.[/bold] Crie um token com permissão de escrita\n"
            "   [cyan]https://huggingface.co/settings/tokens[/cyan]\n\n"
            "[bold]3.[/bold] Abra no Colab, com [bold]T4 GPU[/bold] selecionada:\n"
            "   [cyan]notebooks/colab/01_finetune_qlora_pubmedqa.ipynb[/cyan]\n"
            "   O notebook clona este repositório — o dataset acima já vai junto.\n"
            "   Tempo: 60 a 90 minutos.\n\n"
            "[bold]4.[/bold] Em seguida, no mesmo Colab:\n"
            "   [cyan]notebooks/colab/02_exportar_gguf.ipynb[/cyan]\n"
            "   Funde o adapter, converte para GGUF e publica no Hugging Face Hub.\n\n"
            "[bold]5.[/bold] De volta aqui:\n"
            "   ajuste [cyan]REPO_GGUF_HF[/cyan] no .env e rode "
            "[cyan]make modelo -- --ajustado[/cyan]\n"
            "   depois [cyan]make avaliar[/cyan] — a tabela ganha a coluna do modelo ajustado.",
            title="[bold]Roteiro do treino[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(
        "\n[dim]Enquanto o adapter não existe, o projeto roda com o modelo base "
        "servido sob a mesma persona (medgraph-base) — que é a coluna de referência "
        "que o comparativo precisa de qualquer forma.[/dim]"
    )
    console.print("\n[bold green]Etapa 2 concluída.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
