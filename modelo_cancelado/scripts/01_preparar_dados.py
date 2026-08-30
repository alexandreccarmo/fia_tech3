#!/usr/bin/env python
"""
Etapa 1 — Preparação dos dados.

O QUE ESTE SCRIPT FAZ, NESTA ORDEM:
    1. Baixa o PubMedQA do Hugging Face (pula se já estiver em disco);
    2. Aplica curadoria — filtros de qualidade, anonimização, balanceamento
       e divisão estratificada em treino/validação/teste;
    3. Constrói a base SQLite de prontuários a partir do seed sintético;
    4. Monta o dataset de fine-tuning no formato de conversa;
    5. Apresenta um painel com tudo o que foi produzido.

POR QUE UM ORQUESTRADOR, E NÃO QUATRO COMANDOS SOLTOS:
    As quatro etapas têm dependência estrita entre si e uma ordem que não é
    óbvia de fora. Rodá-las na ordem errada produz erros confusos — montar o
    dataset de fine-tuning antes da curadoria, por exemplo, resulta em um
    arquivo vazio, sem nenhuma mensagem de erro. Um único ponto de entrada
    elimina essa classe de problema.

Uso:
    make dados
    python scripts/01_preparar_dados.py
    python scripts/01_preparar_dados.py --rebaixar     # força novo download
"""

from __future__ import annotations

import json
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
from medgraph.dados import baixar_pubmedqa, construir_banco, curadoria  # noqa: E402
from medgraph.finetune import preparar_dataset_sft  # noqa: E402

console = Console()


def _tabela_arquivos(cfg) -> Table:
    """Inventário do que existe em disco ao final da etapa."""
    tabela = Table(show_header=True, header_style="bold cyan", title="Artefatos produzidos")
    tabela.add_column("Arquivo", width=42)
    tabela.add_column("Registros", justify="right", width=10)
    tabela.add_column("Tamanho", justify="right", width=10)

    def _linhas(caminho: Path) -> int:
        if caminho.suffix == ".jsonl":
            with caminho.open(encoding="utf-8") as f:
                return sum(1 for linha in f if linha.strip())
        return 0

    candidatos = [
        cfg.dir_dados_brutos / "pubmedqa_labeled.jsonl",
        cfg.dir_dados_brutos / "pubmedqa_artificial.jsonl",
        cfg.dir_dados_processados / "pubmedqa_treino.jsonl",
        cfg.dir_dados_processados / "pubmedqa_validacao.jsonl",
        cfg.dir_dados_processados / "pubmedqa_teste.jsonl",
        cfg.dir_dados_processados / "pubmedqa_artificial.jsonl",
        cfg.dir_dados_processados / "sft_train.jsonl",
        cfg.dir_dados_processados / "sft_valid.jsonl",
        cfg.dir_dados_sinteticos / "faq_medicos.jsonl",
        cfg.caminho_banco_prontuarios,
    ]

    for caminho in candidatos:
        if not caminho.exists():
            tabela.add_row(f"[dim]{caminho.name}[/dim]", "[red]ausente[/red]", "-")
            continue
        tamanho = caminho.stat().st_size
        unidade = f"{tamanho / 1024**2:.1f} MB" if tamanho > 1024**2 else f"{tamanho / 1024:.0f} KB"
        registros = _linhas(caminho)
        tabela.add_row(
            str(caminho.relative_to(cfg.dir_raiz)),
            f"{registros:,}" if registros else "-",
            unidade,
        )
    return tabela


def _tabela_corpus_sintetico(cfg) -> Table:
    """O que existe no corpus hospitalar sintético."""
    tabela = Table(show_header=True, header_style="bold cyan", title="Corpus hospitalar sintético")
    tabela.add_column("Fonte", width=30)
    tabela.add_column("Itens", justify="right", width=8)
    tabela.add_column("Observação", overflow="fold")

    protocolos = sorted((cfg.dir_dados_sinteticos / "protocolos").glob("PROT-*.md"))
    tabela.add_row(
        "Protocolos internos",
        str(len(protocolos)),
        ", ".join(p.name.split("-")[1] for p in protocolos[:8]) + ("..." if len(protocolos) > 8 else ""),
    )

    faq = cfg.dir_dados_sinteticos / "faq_medicos.jsonl"
    n_faq = 0
    if faq.exists():
        with faq.open(encoding="utf-8") as f:
            n_faq = sum(1 for linha in f if linha.strip())
    tabela.add_row("FAQ do corpo médico", str(n_faq), "perguntas frequentes em PT-BR")

    docs = sorted((cfg.dir_dados_sinteticos / "modelos_documentos").glob("DOC-*.md"))
    tabela.add_row("Modelos de documentos", str(len(docs)), "laudo, receita, parecer, alta...")

    return tabela


def main() -> int:
    rebaixar = "--rebaixar" in sys.argv

    iniciar(
        banner="Etapa 1 — Preparação dos dados",
        subtitulo="PubMedQA + corpus hospitalar sintético + base de prontuários",
    )
    cfg = obter_settings()

    with abrir_trilha(pergunta="[pipeline] preparação de dados", usuario="sistema") as trilha:
        console.print(Rule("[bold]1/4 · Download do PubMedQA[/bold]"))
        contagem_bruta = baixar_pubmedqa.baixar(cfg, forcar=rebaixar)
        console.print(f"  {contagem_bruta}\n")

        console.print(Rule("[bold]2/4 · Curadoria, anonimização e divisão[/bold]"))
        resumo_curadoria = curadoria.executar(cfg)
        console.print()

        console.print(Rule("[bold]3/4 · Base de prontuários[/bold]"))
        construir_banco.construir(cfg, forcar=True)
        estatisticas = construir_banco.estatisticas(cfg)
        console.print()

        console.print(Rule("[bold]4/4 · Dataset de fine-tuning[/bold]"))
        resumo_sft = preparar_dataset_sft.executar(cfg)
        console.print()

        trace_id = trilha.trace_id

    # -------------------------------------------------------------------------
    # Painel final
    # -------------------------------------------------------------------------
    console.print(Rule("[bold green]Resultado[/bold green]", style="green"))
    console.print(_tabela_corpus_sintetico(cfg))
    console.print()
    console.print(_tabela_arquivos(cfg))
    console.print()

    tabela_base = Table(show_header=True, header_style="bold cyan", title="Base de prontuários")
    tabela_base.add_column("Indicador", width=34)
    tabela_base.add_column("Valor", justify="right", width=10)
    for rotulo, chave in [
        ("Pacientes", "pacientes"),
        ("Idade média (anos)", "idade_media"),
        ("Gestantes", "gestantes"),
        ("Com alguma alergia", "com_alergia"),
        ("Com alergia a penicilina", "com_alergia_penicilina"),
        ("Com exame pendente", "com_exame_pendente"),
        ("Com exame em valor crítico", "com_exame_critico"),
        ("Medicações ativas", "medicacoes_ativas"),
    ]:
        tabela_base.add_row(rotulo, str(estatisticas[chave]))
    console.print(tabela_base)
    console.print()

    tabela_sft = Table(show_header=True, header_style="bold cyan", title="Dataset de fine-tuning")
    tabela_sft.add_column("Rótulo", width=18)
    tabela_sft.add_column("Exemplos", justify="right", width=10)
    tabela_sft.add_column("Proporção", justify="right", width=10)
    for rotulo, qtd in sorted(resumo_sft["por_rotulo"].items(), key=lambda x: -x[1]):
        tabela_sft.add_row(rotulo, f"{qtd:,}", f"{resumo_sft['proporcao_por_rotulo'][rotulo]:.1%}")
    tabela_sft.add_row("[bold]TOTAL[/bold]", f"[bold]{resumo_sft['total']:,}[/bold]", "")
    console.print(tabela_sft)

    console.print(
        f"\n[dim]trace da execução: logs/traces/{trace_id}.json[/dim]"
        f"\n[dim]relatório de curadoria: data/processed/relatorio_curadoria.json[/dim]"
    )

    # Verificação final explícita: o conjunto de teste precisa estar intacto.
    teste = cfg.dir_dados_processados / "pubmedqa_teste.jsonl"
    with teste.open(encoding="utf-8") as f:
        n_teste = sum(1 for linha in f if linha.strip())
    console.print(
        f"\n[bold green]Etapa 1 concluída.[/bold green] "
        f"Conjunto de teste preservado com {n_teste} exemplos, "
        f"[bold]nunca[/bold] utilizados no treino."
    )

    resumo_geral = {
        "bruto": contagem_bruta,
        "curadoria": resumo_curadoria["divisao_anotado"],
        "prontuarios": estatisticas,
        "sft": {k: v for k, v in resumo_sft.items() if k != "por_origem"},
    }
    (cfg.dir_dados_processados / "resumo_etapa1.json").write_text(
        json.dumps(resumo_geral, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
