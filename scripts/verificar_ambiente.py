#!/usr/bin/env python
"""
Diagnostico visual do ambiente do MedGraph.

O QUE FAZ:
    Percorre tudo de que o projeto depende - versao do Python, pacotes
    instalados, arquivo .env, arvore de diretorios, disponibilidade do Ollama,
    presenca dos artefatos de modelo, espaco em disco - e apresenta o
    resultado em uma tabela colorida, item a item.

POR QUE EXISTE:
    Este projeto tem muitas pecas moveis: um modelo treinado em outra maquina
    (Colab), um servidor local (Ollama), uma chave de API opcional e um stack
    de ML pesado. Quando algo nao funciona, a pergunta e sempre "o que esta
    faltando?". Este script responde em cinco segundos, em vez de exigir uma
    caçada por mensagens de erro.

    Tambem e o primeiro comando que o professor deve rodar depois de clonar o
    repositorio - a tabela mostra exatamente o que ja esta pronto e o que
    ainda precisa ser providenciado.

CODIGO DE SAIDA:
    0  ambiente utilizavel (pode haver avisos)
    1  ha ao menos uma falha que impede rodar o projeto

Uso:
    make ambiente
    python scripts/verificar_ambiente.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Permite executar o script direto, sem `pip install -e .`.
RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


# -----------------------------------------------------------------------------
# MODELO DE RESULTADO
# -----------------------------------------------------------------------------
# Tres estados, e nao dois, de proposito:
#   OK      esta tudo certo
#   AVISO   falta algo OPCIONAL ou que sera produzido em uma etapa futura
#           (ex.: o modelo GGUF so existe depois da Etapa 3)
#   FALHA   impede o projeto de rodar agora
# Distinguir AVISO de FALHA evita o efeito "tudo vermelho" logo no inicio,
# que faria o diagnostico perder utilidade.
# -----------------------------------------------------------------------------
OK, AVISO, FALHA = "ok", "aviso", "falha"

_SIMBOLO = {
    OK: "[green]OK[/green]",
    AVISO: "[yellow]AVISO[/yellow]",
    FALHA: "[red]FALHA[/red]",
}


@dataclass
class Verificacao:
    grupo: str
    item: str
    estado: str
    detalhe: str


resultados: list[Verificacao] = []


def checar(grupo: str, item: str, estado: str, detalhe: str = "") -> None:
    resultados.append(Verificacao(grupo, item, estado, detalhe))


# -----------------------------------------------------------------------------
# 1) INTERPRETADOR
# -----------------------------------------------------------------------------
def verificar_python() -> None:
    versao = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] == (3, 12):
        checar("Interpretador", "Versao do Python", OK, versao)
    # noqa abaixo: o ruff considera este ramo morto porque o projeto exige
    # 3.12. Mas justamente por isso ele precisa existir - o diagnostico pode
    # ser executado por um interpretador mais antigo, e a mensagem util e
    # exatamente esta.
    elif sys.version_info[:2] < (3, 12):  # noqa: UP036
        checar("Interpretador", "Versao do Python", FALHA, f"{versao} - o projeto exige 3.12")
    else:
        checar(
            "Interpretador",
            "Versao do Python",
            AVISO,
            f"{versao} - testado em 3.12; wheels de ML podem faltar",
        )

    # Confere, em um processo NOVO, se os pacotes do projeto sao importaveis.
    # Testar no processo atual nao provaria nada: este script ja inseriu os
    # caminhos manualmente no inicio.
    #
    # O teste roda de um diretorio NEUTRO e com PYTHONPATH apontando para o
    # repositorio - que e exatamente como o Makefile executa tudo. Esse e o
    # contrato real do projeto; a instalacao com pip e um extra.
    import os
    import subprocess

    ambiente = {**os.environ, "PYTHONPATH": f"{RAIZ}:{RAIZ / 'src'}"}
    teste = subprocess.run(
        [sys.executable, "-c", "import medgraph, config"],
        capture_output=True,
        text=True,
        cwd=str(Path.home()),
        env=ambiente,
    )
    checar(
        "Interpretador",
        "Pacotes do projeto",
        OK if teste.returncode == 0 else FALHA,
        "medgraph e config importaveis" if teste.returncode == 0
        else teste.stderr.strip().splitlines()[-1],
    )

    # Instalacao com pip: informativo. Quando funciona, o projeto roda de
    # qualquer diretorio sem PYTHONPATH; quando nao, o Makefile cobre.
    sem_pythonpath = subprocess.run(
        [sys.executable, "-c", "import medgraph"],
        capture_output=True,
        text=True,
        cwd=str(Path.home()),
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    checar(
        "Interpretador",
        "Instalacao com pip",
        OK if sem_pythonpath.returncode == 0 else AVISO,
        "ativa - funciona de qualquer diretorio" if sem_pythonpath.returncode == 0
        else "inativa - use os alvos do Makefile, que definem PYTHONPATH",
    )

    em_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checar(
        "Interpretador",
        "Ambiente virtual",
        OK if em_venv else AVISO,
        sys.prefix if em_venv else "rodando fora de um venv",
    )


# -----------------------------------------------------------------------------
# 2) PACOTES
# -----------------------------------------------------------------------------
# Separados por etapa do projeto: assim o diagnostico mostra nao so "falta
# pacote", mas qual parte do trabalho ficaria bloqueada.
# -----------------------------------------------------------------------------
PACOTES = {
    "Configuracao": ["pydantic", "pydantic-settings", "python-dotenv", "PyYAML", "rich"],
    "LangChain": [
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "langchain-ollama",
    ],
    "LangGraph": ["langgraph", "langgraph-checkpoint-sqlite", "grandalf"],
    "RAG": ["faiss-cpu", "sentence-transformers"],
    "Dados": ["datasets", "huggingface-hub", "pandas", "numpy"],
    "Avaliacao": ["scikit-learn", "matplotlib", "tiktoken"],
    "Interface": ["streamlit"],
}


def verificar_pacotes() -> None:
    for grupo, pacotes in PACOTES.items():
        faltando: list[str] = []
        versoes: list[str] = []
        for pacote in pacotes:
            try:
                versoes.append(f"{pacote}=={metadata.version(pacote)}")
            except metadata.PackageNotFoundError:
                faltando.append(pacote)

        if faltando:
            checar(
                "Pacotes",
                grupo,
                FALHA,
                f"faltando: {', '.join(faltando)} - rode: make setup",
            )
        else:
            principal = versoes[0].split("==")
            checar("Pacotes", grupo, OK, f"{len(pacotes)} pacotes | {principal[0]} {principal[1]}")


# -----------------------------------------------------------------------------
# 3) CONFIGURACAO
# -----------------------------------------------------------------------------
def verificar_configuracao() -> None:
    env = RAIZ / ".env"
    if env.exists():
        checar("Configuracao", "Arquivo .env", OK, str(env.relative_to(RAIZ)))
    else:
        checar(
            "Configuracao",
            "Arquivo .env",
            AVISO,
            "nao existe - rode: cp .env.example .env",
        )

    try:
        from config.settings import obter_settings

        cfg = obter_settings()
    except Exception as exc:
        checar("Configuracao", "Leitura das settings", FALHA, f"{type(exc).__name__}: {exc}")
        return

    checar("Configuracao", "Leitura das settings", OK, f"provider={cfg.llm_provider}")
    checar(
        "Configuracao",
        "Politicas de guardrail",
        OK if cfg.caminho_politicas.exists() else FALHA,
        "config/politicas.yaml",
    )

    # A chave da OpenAI e OPCIONAL: o projeto roda 100% offline com o Ollama.
    # Nunca validamos a chave chamando a API - isso custaria dinheiro so para
    # rodar um diagnostico.
    if cfg.openai_configurada:
        checar(
            "Configuracao",
            "Chave da OpenAI",
            OK,
            f"{cfg.resumo_seguro()['openai_api_key']} | teto US$ {cfg.max_custo_usd_sessao:.2f}",
        )
    else:
        checar(
            "Configuracao",
            "Chave da OpenAI",
            AVISO,
            "nao definida - so afeta o comparativo da Etapa 4",
        )

    checar(
        "Configuracao",
        "Token do Hugging Face",
        OK if cfg.hf_token else AVISO,
        "necessario apenas para baixar o Llama 3.2 (modelo gated)",
    )


# -----------------------------------------------------------------------------
# 4) ESTRUTURA DE PASTAS
# -----------------------------------------------------------------------------
def verificar_diretorios() -> None:
    try:
        from config.settings import obter_settings

        cfg = obter_settings()
    except Exception:
        return

    esperados = {
        "data/": cfg.dir_dados,
        "logs/": cfg.dir_logs,
        "logs/auditoria/": cfg.dir_auditoria,
        "logs/traces/": cfg.dir_traces,
        "models/adapters/": cfg.dir_adapters,
        "docs/graficos/": cfg.dir_graficos,
    }
    ausentes = [nome for nome, caminho in esperados.items() if not caminho.is_dir()]
    if ausentes:
        checar("Estrutura", "Arvore de diretorios", AVISO, f"ausentes: {', '.join(ausentes)}")
    else:
        checar("Estrutura", "Arvore de diretorios", OK, f"{len(esperados)} diretorios presentes")


# -----------------------------------------------------------------------------
# 5) ARTEFATOS PRODUZIDOS PELAS ETAPAS SEGUINTES
# -----------------------------------------------------------------------------
# Sao AVISO e nao FALHA: na Etapa 0 nenhum deles existe ainda, e isso e o
# esperado. A tabela funciona aqui como um painel de progresso do projeto.
# -----------------------------------------------------------------------------
def verificar_artefatos() -> None:
    try:
        from config.settings import obter_settings

        cfg = obter_settings()
    except Exception:
        return

    dataset = cfg.dir_dados_processados / "sft_train.jsonl"
    checar(
        "Artefatos",
        "Dataset de fine-tuning",
        OK if dataset.exists() else AVISO,
        str(dataset.relative_to(RAIZ)) if dataset.exists() else "gerado na Etapa 1/2",
    )

    adapters = list(cfg.dir_adapters.glob("*/adapter_model.safetensors"))
    checar(
        "Artefatos",
        "Adapter LoRA",
        OK if adapters else AVISO,
        str(adapters[0].parent.name) if adapters else "produzido no Colab, na Etapa 2",
    )

    ggufs = list(cfg.dir_gguf.glob("*.gguf"))
    if ggufs:
        tamanho_gb = ggufs[0].stat().st_size / 1024**3
        checar("Artefatos", "Modelo GGUF", OK, f"{ggufs[0].name} ({tamanho_gb:.1f} GB)")
    else:
        checar("Artefatos", "Modelo GGUF", AVISO, "baixado na Etapa 3 (make modelo)")

    indices = list(cfg.dir_indices.glob("*/index.faiss"))
    checar(
        "Artefatos",
        "Indice FAISS",
        OK if indices else AVISO,
        str(indices[0].parent.name) if indices else "construido na Etapa 5 (make indexar)",
    )

    checar(
        "Artefatos",
        "Base de prontuarios",
        OK if cfg.caminho_banco_prontuarios.exists() else AVISO,
        "prontuarios.sqlite" if cfg.caminho_banco_prontuarios.exists() else "criada na Etapa 6",
    )


# -----------------------------------------------------------------------------
# 6) SERVICOS EXTERNOS
# -----------------------------------------------------------------------------
def verificar_ollama() -> None:
    """
    Confere se o Ollama esta no ar e se o modelo do projeto ja foi registrado.

    Nao e erro estar ausente na Etapa 0: o modelo so passa a existir depois
    do fine-tuning. Mas e a dependencia que mais causa confusao no dia da
    apresentacao, entao aparece com destaque na tabela.
    """
    try:
        from config.settings import obter_settings

        cfg = obter_settings()
    except Exception:
        return

    if shutil.which("ollama") is None:
        checar("Servicos", "Ollama instalado", AVISO, "nao encontrado - brew install ollama")
        return
    checar("Servicos", "Ollama instalado", OK, shutil.which("ollama") or "")

    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=3) as resposta:
            modelos = [m["name"] for m in json.loads(resposta.read()).get("models", [])]
    except Exception as exc:
        checar(
            "Servicos",
            "Ollama no ar",
            AVISO,
            f"{cfg.ollama_base_url} inacessivel ({type(exc).__name__}) - brew services start ollama",
        )
        return

    checar("Servicos", "Ollama no ar", OK, f"{len(modelos)} modelo(s) registrado(s)")

    tem_modelo = any(m.split(":")[0] == cfg.ollama_model for m in modelos)
    checar(
        "Servicos",
        f"Modelo '{cfg.ollama_model}'",
        OK if tem_modelo else AVISO,
        "registrado" if tem_modelo else "sera registrado na Etapa 3 (make modelo)",
    )


def verificar_disco() -> None:
    uso = shutil.disk_usage(RAIZ)
    livre_gb = uso.free / 1024**3
    # O projeto completo ocupa cerca de 12 GB: torch (~2,5), modelo base no
    # cache do HF (~7), GGUF (~2) e datasets (~0,5).
    if livre_gb >= 20:
        checar("Recursos", "Espaco em disco", OK, f"{livre_gb:.1f} GB livres")
    elif livre_gb >= 12:
        checar("Recursos", "Espaco em disco", AVISO, f"{livre_gb:.1f} GB livres - justo")
    else:
        checar("Recursos", "Espaco em disco", FALHA, f"{livre_gb:.1f} GB livres - precisa de ~12 GB")


# -----------------------------------------------------------------------------
# APRESENTACAO
# -----------------------------------------------------------------------------
def main() -> int:
    console.print(
        Panel(
            "[bold]Diagnostico do ambiente[/bold]\n"
            "[dim]MedGraph - Tech Challenge Fase 3 - 8IADT[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    verificar_python()
    verificar_pacotes()
    verificar_configuracao()
    verificar_diretorios()
    verificar_artefatos()
    verificar_ollama()
    verificar_disco()

    tabela = Table(show_header=True, header_style="bold cyan", expand=True)
    tabela.add_column("Grupo", style="dim", width=15)
    tabela.add_column("Item", width=26)
    tabela.add_column("Estado", justify="center", width=8)
    tabela.add_column("Detalhe", overflow="fold")

    grupo_anterior = None
    for r in resultados:
        tabela.add_row(
            r.grupo if r.grupo != grupo_anterior else "",
            r.item,
            _SIMBOLO[r.estado],
            r.detalhe,
        )
        grupo_anterior = r.grupo

    console.print(tabela)

    falhas = [r for r in resultados if r.estado == FALHA]
    avisos = [r for r in resultados if r.estado == AVISO]

    console.print(
        f"\n[green]{len(resultados) - len(falhas) - len(avisos)} ok[/green]  "
        f"[yellow]{len(avisos)} aviso(s)[/yellow]  "
        f"[red]{len(falhas)} falha(s)[/red]"
    )

    if falhas:
        console.print("\n[bold red]O ambiente NAO esta pronto. Resolva:[/bold red]")
        for r in falhas:
            console.print(f"  [red]x[/red] {r.grupo} / {r.item}: {r.detalhe}")
        return 1

    console.print(
        "\n[bold green]Ambiente utilizavel.[/bold green] "
        "[dim]Avisos indicam artefatos que serao produzidos nas proximas etapas.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
