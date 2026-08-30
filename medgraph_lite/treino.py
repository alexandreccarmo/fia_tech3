"""
Fine-tuning por QLoRA, dimensionado para caber numa apresentacao.

POR QUE UM MODELO DE 0,5 BILHAO:
    O enunciado deixa a escolha livre ("como LLaMA, Falcon ou um outro"). Um
    modelo de 3B leva quase 6 horas numa T4 gratuita - acima da cota diaria do
    Colab, e impossivel de demonstrar ao vivo. O Qwen2.5-0.5B treina em ~8
    minutos e demonstra exatamente a mesma tecnica.

    O objetivo do ajuste tambem nao exige um modelo grande: nao estamos
    ensinando medicina, e sim o FORMATO da resposta - decisao na primeira
    linha, fonte citada no fim. E o formato que os guardrails precisam
    encontrar para verificar.

POR QUE QLoRA:
    O modelo base entra quantizado em 4 bits e congelado; treinam-se apenas
    matrizes de baixo posto ao lado das camadas de atencao e do MLP. Sao ~1% dos
    parametros, e o artefato final tem alguns megabytes em vez de gigabytes.
"""

from __future__ import annotations

from typing import Any

MODELO_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

SISTEMA = (
    "Voce e um assistente clinico do Hospital Vida Plena. Responda SEMPRE neste "
    "formato:\n"
    "Decisao: yes|no|maybe\n"
    "<justificativa em ate 3 frases, apoiada apenas no contexto fornecido>\n"
    "Cite a fonte entre colchetes, por exemplo [P1] ou [E1].\n"
    "Nunca prescreva: apresente a evidencia e devolva a decisao ao medico."
)

CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # As sete projecoes: atencao (q,k,v,o) e MLP (gate,up,down). Aplicar so em
    # q e v, como e comum em tutoriais, rende menos quando a tarefa muda o
    # ESTILO da resposta - que e exatamente o nosso caso.
    "alvos": ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"],
    "max_seq_length": 512,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "epocas": 1,
    "learning_rate": 2e-4,
}


def formatar(exemplo: dict) -> str:
    """Monta o texto de treino no formato de conversa do modelo."""
    return (
        f"<|im_start|>system\n{SISTEMA}<|im_end|>\n"
        f"<|im_start|>user\nContexto:\n{exemplo['contexto']}\n\n"
        f"Pergunta: {exemplo['pergunta']}<|im_end|>\n"
        f"<|im_start|>assistant\n{exemplo['resposta']}<|im_end|>"
    )


def config_quantizacao(suporta_bf16: bool):
    """A parte 'Q' do QLoRA: 4 bits em NF4, com dupla quantizacao."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if suporta_bf16 else torch.float16,
    )


def config_lora():
    from peft import LoraConfig

    return LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["alvos"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def argumentos_de_treino(saida: str, suporta_bf16: bool) -> dict[str, Any]:
    """
    Argumentos do treinador, num dicionario simples.

    Sao entregues como dicionario, e nao como objeto pronto, porque a API do
    `trl` renomeia argumentos entre versoes. O notebook monta a configuracao
    tentando e removendo o que a versao instalada recusar - assim uma mudanca
    de nome vira um aviso, e nao um treino que nao roda.
    """
    return {
        "output_dir": saida,
        "per_device_train_batch_size": CONFIG["batch_size"],
        "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
        "num_train_epochs": CONFIG["epocas"],
        "learning_rate": CONFIG["learning_rate"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 2,
        "save_strategy": "no",
        "report_to": "none",
        "bf16": suporta_bf16,
        "fp16": not suporta_bf16,
        "max_seq_length": CONFIG["max_seq_length"],
        "max_length": CONFIG["max_seq_length"],
        "packing": False,
        "seed": 42,
    }


def construir_tolerante(classe, argumentos: dict[str, Any]):
    """
    Instancia a classe descartando apenas o que ela mesma recusar.

    Descobrir por introspeccao quais argumentos a classe aceita erra para menos
    - e errar para menos aqui significa treinar com uma configuracao diferente
    da pedida, em silencio. Tentar e recuar deixa a propria biblioteca decidir.
    """
    import re

    restantes = dict(argumentos)
    descartados: list[str] = []
    padrao = re.compile(r"unexpected keyword argument '([^']+)'")

    for _ in range(len(argumentos) + 1):
        try:
            return classe(**restantes), sorted(descartados)
        except TypeError as erro:
            achado = padrao.search(str(erro))
            if achado is None or achado.group(1) not in restantes:
                raise
            del restantes[achado.group(1)]
            descartados.append(achado.group(1))
    raise RuntimeError(f"nao foi possivel instanciar {classe.__name__}")
