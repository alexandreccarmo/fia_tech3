#!/usr/bin/env python
"""
Etapa 3 — Registro do modelo no Ollama.

O QUE ESTE SCRIPT FAZ:
    1. Regenera o `Modelfile` a partir de `src/medgraph/chains/prompts.py`,
       garantindo que o prompt de sistema servido pelo Ollama seja o mesmo
       usado no fine-tuning e nas chains;
    2. Localiza os pesos — o GGUF ajustado, baixando-o do Hugging Face Hub se
       necessário, ou o modelo base quando o fine-tuning ainda não foi feito;
    3. Registra o modelo no Ollama com o nome `medgraph`;
    4. Faz uma chamada de verificação e mostra a resposta.

OS DOIS MODOS DE OPERAÇÃO

    --ajustado  (padrão)
        Usa o GGUF produzido no Colab. É o modelo do projeto.

    --base
        Registra o modelo BASE com a mesma persona e os mesmos parâmetros,
        sob o nome `medgraph-base`.

        Isso não é um atalho: é uma peça necessária da avaliação. O
        comparativo da Etapa 4 mede o ganho do fine-tuning, e para isso
        precisa dos dois modelos servidos em condições idênticas — mesmo
        prompt de sistema, mesma temperatura, mesmo template. Qualquer
        diferença nesses parâmetros contaminaria a comparação e atribuiria
        ao treino um ganho que na verdade veio da configuração.

        Também é o que permite executar o projeto de ponta a ponta antes de
        o fine-tuning estar pronto.

POR QUE REGENERAR O MODELFILE EM VEZ DE MANTÊ-LO À MÃO:
    O prompt de sistema aparece em três lugares — no dataset de treino, nas
    chains do LangChain e no modelo registrado no Ollama. Mantê-lo copiado à
    mão nos três garante que divirjam. Gerando o Modelfile a partir do módulo
    de prompts, a divergência deixa de ser possível.

Uso:
    make modelo
    python scripts/03_instalar_modelo.py --base        # sem fine-tuning
    python scripts/03_instalar_modelo.py --ajustado    # com o GGUF do Colab
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402

from config.settings import obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.chains import prompts  # noqa: E402
from medgraph.logging_config import obter_logger  # noqa: E402

console = Console()
log = obter_logger(__name__)

# Sequencias de escape ANSI emitidas pelo streaming do `ollama run`.
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")

# Modelo base público usado quando o fine-tuning ainda não foi executado.
MODELO_BASE_OLLAMA = "llama3.2:3b"


def gerar_modelfile(origem_pesos: str, nome_modelo: str) -> str:
    """
    Monta o conteúdo do Modelfile a partir do módulo de prompts.

    Args:
        origem_pesos: valor da diretiva FROM — caminho de um .gguf ou nome de
            um modelo já presente no Ollama.
    """
    return f'''# =============================================================================
# MedGraph — receita de registro no Ollama
#
# ARQUIVO GERADO por scripts/03_instalar_modelo.py. Não edite à mão: o prompt
# de sistema abaixo é uma cópia literal de src/medgraph/chains/prompts.py, e
# precisa continuar sendo. Ele é o MESMO prompt usado no fine-tuning e nas
# chains do LangChain — divergência entre os três degrada a resposta de forma
# silenciosa, sem nenhum erro visível.
#
# Registrar com:  ollama create {nome_modelo} -f Modelfile
# =============================================================================

FROM {origem_pesos}

# -----------------------------------------------------------------------------
# PROMPT DE SISTEMA — limites de atuação do assistente  [REQ-3a]
# -----------------------------------------------------------------------------
# Fica registrado junto ao modelo de propósito: qualquer cliente que chame
# `{nome_modelo}`, inclusive um `ollama run` no terminal, recebe o assistente
# com as suas regras de segurança, e não o modelo cru.
# -----------------------------------------------------------------------------
SYSTEM """{prompts.SISTEMA}"""

# -----------------------------------------------------------------------------
# PARÂMETROS DE GERAÇÃO
# -----------------------------------------------------------------------------
# Temperatura baixa é decisão clínica, não de engenharia: a mesma pergunta
# sobre o mesmo paciente precisa produzir a mesma conduta. Um assistente que
# varia a resposta a cada consulta é inauditável.
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05

# 4096 tokens acomodam o prompt de sistema, os quatro trechos do RAG e o
# resumo do prontuário, com margem para a resposta.
PARAMETER num_ctx 4096
'''


def ollama_disponivel() -> bool:
    return shutil.which("ollama") is not None


def modelos_registrados() -> list[str]:
    resultado = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=False
    )
    return [
        linha.split()[0]
        for linha in resultado.stdout.splitlines()[1:]
        if linha.strip()
    ]


def garantir_modelo_base() -> bool:
    """Baixa o modelo base do Ollama se ainda não estiver presente."""
    if any(m.split(":")[0] == MODELO_BASE_OLLAMA.split(":")[0] for m in modelos_registrados()):
        console.print(f"  [green]ok[/green] {MODELO_BASE_OLLAMA} já disponível")
        return True

    console.print(f"  baixando {MODELO_BASE_OLLAMA} (~2 GB)...")
    resultado = subprocess.run(["ollama", "pull", MODELO_BASE_OLLAMA], check=False)
    return resultado.returncode == 0


def baixar_gguf(cfg) -> Path | None:
    """Recupera o GGUF ajustado do Hugging Face Hub."""
    destino = cfg.dir_gguf / cfg.arquivo_gguf
    if destino.exists():
        console.print(f"  [green]ok[/green] GGUF já em disco ({destino.stat().st_size / 1024**3:.1f} GB)")
        return destino

    console.print(f"  baixando {cfg.arquivo_gguf} de {cfg.repo_gguf_hf} (~2 GB)...")
    try:
        from huggingface_hub import hf_hub_download

        caminho = hf_hub_download(
            repo_id=cfg.repo_gguf_hf,
            filename=cfg.arquivo_gguf,
            local_dir=str(cfg.dir_gguf),
            token=cfg.hf_token or None,
        )
        return Path(caminho)
    except Exception as exc:
        console.print(f"  [red]falhou[/red]: {type(exc).__name__}: {exc}")
        return None


def registrar(nome_modelo: str, modelfile: str, diretorio: Path) -> bool:
    """Grava o Modelfile e chama `ollama create`."""
    caminho = diretorio / "Modelfile"
    caminho.write_text(modelfile, encoding="utf-8")
    console.print(f"  Modelfile gravado em {caminho.relative_to(RAIZ)}")

    resultado = subprocess.run(
        ["ollama", "create", nome_modelo, "-f", str(caminho)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(diretorio),
    )
    if resultado.returncode != 0:
        console.print(f"  [red]ollama create falhou[/red]:\n{resultado.stderr}")
        return False
    console.print(f"  [green]ok[/green] modelo '{nome_modelo}' registrado")
    return True


def verificar(nome_modelo: str) -> bool:
    """Faz uma pergunta clínica de teste e mostra a resposta."""
    pergunta = (
        "Um paciente adulto chega ao pronto-socorro com suspeita de sepse. "
        "Em uma frase, qual é a prioridade da primeira hora?"
    )
    console.print(f"\n  [dim]pergunta de teste:[/dim] {pergunta}")

    resultado = subprocess.run(
        ["ollama", "run", nome_modelo, pergunta],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if resultado.returncode != 0:
        console.print(f"  [red]falhou[/red]: {resultado.stderr[:400]}")
        return False

    # O `ollama run` transmite a resposta token a token e usa sequencias ANSI
    # de controle de cursor para reescrever a linha. Capturadas em um pipe,
    # essas sequencias viram lixo visivel no meio do texto. Removemos.
    limpo = ANSI.sub("", resultado.stdout).strip()

    console.print(
        Panel(
            limpo or "(resposta vazia)",
            title=f"[bold]resposta de {nome_modelo}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    return True


def main() -> int:
    analisador = argparse.ArgumentParser(description="Registra o modelo do MedGraph no Ollama.")
    grupo = analisador.add_mutually_exclusive_group()
    grupo.add_argument("--ajustado", action="store_true", help="usa o GGUF ajustado (padrão)")
    grupo.add_argument("--base", action="store_true", help="registra o modelo base como medgraph-base")
    grupo.add_argument("--ambos", action="store_true", help="registra os dois, para o comparativo")
    argumentos = analisador.parse_args()

    iniciar(banner="Etapa 3 — Registro do modelo no Ollama")
    cfg = obter_settings()

    if not ollama_disponivel():
        console.print(
            "[red]Ollama não encontrado.[/red]\n"
            "  macOS:  brew install ollama && brew services start ollama\n"
            "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
        )
        return 1

    fazer_base = argumentos.base or argumentos.ambos or not argumentos.ajustado
    fazer_ajustado = argumentos.ajustado or argumentos.ambos

    sucesso = True
    diretorio_finetune = RAIZ / "src" / "medgraph" / "finetune"

    # -------------------------------------------------------------------------
    if fazer_base:
        console.print(Rule("[bold]Modelo base (referência do comparativo)[/bold]"))
        if not garantir_modelo_base():
            console.print("[red]não foi possível obter o modelo base[/red]")
            return 1

        modelfile = gerar_modelfile(MODELO_BASE_OLLAMA, "medgraph-base")
        if registrar("medgraph-base", modelfile, diretorio_finetune):
            verificar("medgraph-base")
        else:
            sucesso = False

    # -------------------------------------------------------------------------
    if fazer_ajustado:
        console.print(Rule("[bold]Modelo ajustado (fine-tuning)[/bold]"))
        gguf = baixar_gguf(cfg)
        if gguf is None:
            console.print(
                "\n[yellow]O modelo ajustado ainda não está disponível.[/yellow]\n"
                "  Para produzi-lo, execute no Google Colab, nesta ordem:\n"
                "    notebooks/colab/01_finetune_qlora_pubmedqa.ipynb\n"
                "    notebooks/colab/02_exportar_gguf.ipynb\n"
                "  Depois ajuste REPO_GGUF_HF no .env e rode novamente.\n"
            )
        else:
            modelfile = gerar_modelfile(f"./{gguf.name}", cfg.ollama_model)
            if registrar(cfg.ollama_model, modelfile, gguf.parent):
                verificar(cfg.ollama_model)
            else:
                sucesso = False

    # -------------------------------------------------------------------------
    console.print(Rule(style="green" if sucesso else "red"))
    registrados = modelos_registrados()
    console.print(f"Modelos no Ollama: {registrados}")

    tem_ajustado = any(m.split(":")[0] == cfg.ollama_model for m in registrados)
    if not tem_ajustado:
        console.print(
            f"\n[yellow]Atenção:[/yellow] o modelo '{cfg.ollama_model}' (ajustado) ainda não existe.\n"
            f"Para que o projeto rode agora, defina no .env:\n"
            f"  [cyan]OLLAMA_MODEL=medgraph-base[/cyan]\n"
            f"e troque de volta para [cyan]medgraph[/cyan] depois do fine-tuning."
        )

    return 0 if sucesso else 1


if __name__ == "__main__":
    raise SystemExit(main())
