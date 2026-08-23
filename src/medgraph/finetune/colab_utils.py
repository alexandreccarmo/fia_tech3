"""
[REQ-1] Utilidades do fine-tuning executado no Google Colab.

O QUE FAZ:
    Concentra a logica que o notebook do Colab precisa mas que nao deveria
    viver dentro de uma celula: verificacao do ambiente de GPU, montagem da
    configuracao de QLoRA, compatibilidade entre versoes de biblioteca e
    geracao do grafico de perda.

POR QUE CODIGO EM MODULO E NAO EM CELULA:
    Codigo dentro de notebook nao e testavel, nao e revisavel em diff e
    tende a divergir da versao "boa" que alguem rodou uma vez. Deixando a
    logica aqui, o notebook fica sendo o que deve ser - um roteiro de
    execucao legivel - e o que ele executa continua sob controle de versao
    como codigo normal.

A COMPATIBILIDADE DE VERSOES E O PONTO MAIS FRAGIL:
    A biblioteca `trl` mudou a assinatura do SFTTrainer varias vezes em pouco
    tempo: `TrainingArguments` virou `SFTConfig`; `max_seq_length` e
    `dataset_text_field` migraram de argumento do treinador para campo da
    configuracao; `tokenizer` foi renomeado para `processing_class`. Um
    notebook que fixe uma assinatura quebra na proxima versao que o Colab
    instalar.

    A funcao `montar_treinador` inspeciona a assinatura em tempo de execucao
    e passa apenas o que aquela versao aceita. E mais codigo do que a chamada
    direta, mas e a diferenca entre um notebook que roda daqui a seis meses e
    um que exige arqueologia de versoes.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# HIPERPARAMETROS
# -----------------------------------------------------------------------------
# Cada valor abaixo foi escolhido para caber na GPU T4 de 16 GB do Colab
# gratuito, treinando um modelo de 3 bilhoes de parametros em 4 bits.
# -----------------------------------------------------------------------------
CONFIG_PADRAO: dict[str, Any] = {
    # --- LoRA -------------------------------------------------------------
    # r=16 e o ponto de equilibrio usual: r=8 costuma subajustar em tarefas
    # que exigem formato de saida rigido (como a nossa, com "Decisão:" na
    # primeira linha); r=32 dobra os parametros treinaveis sem ganho claro
    # nesta escala de dados.
    "lora_r": 16,
    # alpha = 2 x r e a convencao mais comum; o fator de escala efetivo
    # (alpha/r) fica em 2.
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Aplicar LoRA nas sete projecoes - atencao E MLP - rende bem melhor do
    # que so em q_proj/v_proj quando a tarefa muda o ESTILO da resposta, e
    # nao apenas o conhecimento.
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # --- Treino -----------------------------------------------------------
    # 1024 tokens cobre a grande maioria dos nossos exemplos (media ~900).
    # Subir para 2048 dobraria a memoria de ativacao e nao caberia na T4.
    "max_seq_length": 1024,
    # Lote fisico 2 + acumulo 8 = lote efetivo 16. O lote fisico e o maximo
    # que a T4 aguenta nesta sequencia; o acumulo recupera a estabilidade do
    # gradiente sem custo de memoria.
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 2,
    # 2e-4 e a taxa de referencia para LoRA. Uma ordem de grandeza acima do
    # que se usa em fine-tuning completo, porque so os adaptadores treinam.
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 2,
    "seed": 42,
}


# =============================================================================
# AMBIENTE
# =============================================================================
def verificar_gpu() -> dict[str, Any]:
    """
    Descreve a GPU disponivel e avisa quando o ambiente nao serve.

    Falhar aqui, no primeiro minuto, e muito melhor do que descobrir a
    ausencia de GPU depois de trinta minutos de download do modelo.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nenhuma GPU disponivel.\n"
            "No Colab: menu Ambiente de execucao > Alterar o tipo de ambiente "
            "de execucao > Acelerador de hardware = T4 GPU."
        )

    propriedades = torch.cuda.get_device_properties(0)
    memoria_gb = propriedades.total_memory / 1024**3
    info = {
        "nome": propriedades.name,
        "memoria_gb": round(memoria_gb, 1),
        "capacidade": f"{propriedades.major}.{propriedades.minor}",
        # bfloat16 exige capacidade de computo 8.0 ou superior (Ampere).
        # A T4 e Turing (7.5) e so tem float16 - o que muda o dtype de treino.
        "suporta_bf16": propriedades.major >= 8,
        "torch": torch.__version__,
    }

    if memoria_gb < 14:
        print(
            f"AVISO: {memoria_gb:.1f} GB de VRAM. O treino foi dimensionado para 16 GB "
            f"(T4). Reduza max_seq_length para 768 se ocorrer erro de memoria."
        )
    return info


def versoes_instaladas() -> dict[str, str]:
    """
    Retrato das versoes das bibliotecas criticas.

    Vai para o relatorio tecnico e para o cabecalho do treino: quando alguem
    tentar reproduzir o resultado meses depois, esta tabela e o que permite
    reconstruir o ambiente.
    """
    import importlib.metadata as metadata

    pacotes = (
        "torch", "transformers", "trl", "peft",
        "bitsandbytes", "accelerate", "datasets",
    )
    versoes: dict[str, str] = {}
    for pacote in pacotes:
        try:
            versoes[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versoes[pacote] = "(nao instalado)"
    return versoes


# =============================================================================
# MODELO
# =============================================================================
def config_quantizacao(suporta_bf16: bool):
    """
    Configuracao de carregamento em 4 bits (a parte "Q" do QLoRA).

    NF4 (NormalFloat4) e o tipo de dado proposto no artigo do QLoRA: e
    otimizado para pesos que seguem distribuicao normal, que e o caso dos
    pesos de uma rede treinada. Perde menos qualidade que o int4 comum.

    A dupla quantizacao comprime tambem as CONSTANTES de quantizacao,
    economizando cerca de 0,4 bit por parametro - relevante quando a margem
    de VRAM e estreita.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        # O calculo acontece em precisao maior que o armazenamento. Na T4,
        # que nao tem bfloat16, usamos float16.
        bnb_4bit_compute_dtype=torch.bfloat16 if suporta_bf16 else torch.float16,
    )


def config_lora(cfg: dict[str, Any] | None = None):
    """Configuracao dos adaptadores LoRA."""
    from peft import LoraConfig

    cfg = {**CONFIG_PADRAO, **(cfg or {})}
    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


# =============================================================================
# TREINADOR — camada de compatibilidade
# =============================================================================
def _filtrar_argumentos(classe, argumentos: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Mantem apenas os argumentos que a assinatura da classe realmente aceita.

    Devolve tambem os descartados, para que o notebook os IMPRIMA. Um
    argumento silenciosamente ignorado e pior do que um erro: o treino roda
    com uma configuracao diferente da pretendida e ninguem percebe.
    """
    try:
        parametros = set(inspect.signature(classe.__init__).parameters)
    except (TypeError, ValueError):
        return argumentos, []

    aceitos = {k: v for k, v in argumentos.items() if k in parametros}
    descartados = sorted(set(argumentos) - set(aceitos))
    return aceitos, descartados


def montar_configuracao_sft(diretorio_saida: str, cfg: dict[str, Any] | None = None):
    """
    Monta o objeto de configuracao do SFTTrainer da versao instalada.

    Ver a nota sobre compatibilidade no cabecalho do modulo.
    """
    from trl import SFTConfig

    cfg = {**CONFIG_PADRAO, **(cfg or {})}
    gpu = verificar_gpu()

    desejados: dict[str, Any] = {
        "output_dir": diretorio_saida,
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "num_train_epochs": cfg["num_train_epochs"],
        "learning_rate": cfg["learning_rate"],
        "lr_scheduler_type": cfg["lr_scheduler_type"],
        "warmup_ratio": cfg["warmup_ratio"],
        "weight_decay": cfg["weight_decay"],
        "max_grad_norm": cfg["max_grad_norm"],
        "optim": cfg["optim"],
        "logging_steps": cfg["logging_steps"],
        "save_steps": cfg["save_steps"],
        "eval_steps": cfg["eval_steps"],
        "save_total_limit": cfg["save_total_limit"],
        "seed": cfg["seed"],
        "max_seq_length": cfg["max_seq_length"],
        "max_length": cfg["max_seq_length"],  # nome usado em versoes recentes
        "packing": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "bf16": gpu["suporta_bf16"],
        "fp16": not gpu["suporta_bf16"],
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    aceitos, descartados = _filtrar_argumentos(SFTConfig, desejados)
    if descartados:
        print(
            "Argumentos nao suportados por esta versao do trl e portanto "
            f"IGNORADOS: {descartados}"
        )
    return SFTConfig(**aceitos)


def montar_treinador(modelo, tokenizador, dados_treino, dados_validacao, configuracao, lora):
    """Instancia o SFTTrainer lidando com as renomeacoes de argumento."""
    from trl import SFTTrainer

    desejados: dict[str, Any] = {
        "model": modelo,
        "args": configuracao,
        "train_dataset": dados_treino,
        "eval_dataset": dados_validacao,
        "peft_config": lora,
        "processing_class": tokenizador,  # nome atual
        "tokenizer": tokenizador,         # nome antigo
    }
    aceitos, descartados = _filtrar_argumentos(SFTTrainer, desejados)

    # `tokenizer` e `processing_class` sao o mesmo argumento com nomes de
    # epocas diferentes. Passar os dois derrubaria a chamada.
    if "processing_class" in aceitos:
        aceitos.pop("tokenizer", None)

    if descartados:
        print(f"Argumentos do SFTTrainer ignorados nesta versao: {descartados}")
    return SFTTrainer(**aceitos)


# =============================================================================
# RESULTADOS
# =============================================================================
def grafico_de_perda(historico: list[dict[str, Any]], destino: str | Path) -> str:
    """
    Desenha a curva de perda de treino e de validacao.

    E o grafico que responde a pergunta "o treino funcionou?" sem ambiguidade:
    perda de treino caindo e de validacao subindo significa sobreajuste;
    ambas paradas significam taxa de aprendizado baixa demais ou dados
    insuficientes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passos_treino = [(h["step"], h["loss"]) for h in historico if "loss" in h]
    passos_eval = [(h["step"], h["eval_loss"]) for h in historico if "eval_loss" in h]

    figura, eixo = plt.subplots(figsize=(9, 5))
    if passos_treino:
        eixo.plot(*zip(*passos_treino, strict=False), label="perda de treino", linewidth=1.6)
    if passos_eval:
        eixo.plot(
            *zip(*passos_eval, strict=False),
            label="perda de validação",
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    eixo.set_xlabel("passo de treino")
    eixo.set_ylabel("perda (cross-entropy)")
    eixo.set_title("Fine-tuning QLoRA — MedGraph")
    eixo.legend()
    eixo.grid(alpha=0.3)
    figura.tight_layout()

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return str(destino)


def salvar_metadados(
    destino: str | Path,
    *,
    modelo_base: str,
    configuracao: dict[str, Any],
    versoes: dict[str, str],
    gpu: dict[str, Any],
    historico: list[dict[str, Any]],
    exemplos_treino: int,
    exemplos_validacao: int,
    duracao_s: float,
) -> str:
    """
    Grava um cartao de treino ao lado do adapter.

    Sem isso, o adapter e um arquivo binario sem procedencia. Com isso, e
    possivel responder meses depois: qual modelo base, com que dados, quais
    hiperparametros, em qual GPU, com quais versoes, em quanto tempo, e com
    que perda final.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    perdas_eval = [h["eval_loss"] for h in historico if "eval_loss" in h]
    perdas_treino = [h["loss"] for h in historico if "loss" in h]

    metadados = {
        "modelo_base": modelo_base,
        "metodo": "QLoRA (NF4, dupla quantizacao)",
        "hiperparametros": configuracao,
        "gpu": gpu,
        "versoes": versoes,
        "dados": {"treino": exemplos_treino, "validacao": exemplos_validacao},
        "duracao_segundos": round(duracao_s, 1),
        "duracao_legivel": f"{duracao_s / 60:.1f} min",
        "perda_treino_inicial": perdas_treino[0] if perdas_treino else None,
        "perda_treino_final": perdas_treino[-1] if perdas_treino else None,
        "perda_validacao_inicial": perdas_eval[0] if perdas_eval else None,
        "perda_validacao_final": perdas_eval[-1] if perdas_eval else None,
        "historico": historico,
    }
    destino.write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destino)
