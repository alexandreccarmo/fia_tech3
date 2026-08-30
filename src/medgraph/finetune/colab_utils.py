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

import contextlib
import dataclasses
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
        # Sao duas causas distintas, e a mensagem precisa separar as duas.
        # Mandar trocar o acelerador quando o problema e cota faz procurar o
        # defeito num menu que ja esta correto - e a cota e a causa mais
        # provavel justamente para quem ja treinou algumas horas no dia.
        raise RuntimeError(
            "Nenhuma GPU disponivel neste ambiente.\n\n"
            "1. RUNTIME SEM ACELERADOR\n"
            "   Ambiente de execucao > Alterar o tipo de ambiente de execucao\n"
            "   > Acelerador de hardware = T4 GPU. Depois rode de novo a\n"
            "   partir da celula que clona o repositorio.\n\n"
            "2. COTA DIARIA DE GPU ESGOTADA\n"
            "   Provavel se voce ja treinou algumas horas hoje. O Colab\n"
            "   gratuito nao recusa a conexao: ele conecta em CPU e mantem\n"
            "   'T4 GPU' selecionada no menu, o que faz o caso 1 parecer\n"
            "   resolvido quando nao e.\n"
            "   Para confirmar, rode  !nvidia-smi  -- se o comando nao\n"
            "   existir, e cota.\n"
            "   A cota volta sozinha em algumas horas e nao ha como consultar\n"
            "   quanto falta. Alternativas: outra conta Google, ou Colab Pro."
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
def _nomes_aceitos(classe) -> set[str]:
    """
    Descobre todos os nomes de parametro que a classe aceita.

    POR QUE NAO BASTA inspecionar `__init__`:
        `SFTConfig` e uma dataclass que herda de `TrainingArguments`. Quando o
        `__init__` gerado usa `**kwargs`, ou quando a hierarquia e montada por
        composicao, `inspect.signature(classe.__init__)` nao enxerga os campos
        herdados — e o filtro descarta configuracao valida.

        Foi o que aconteceu num treino real: `warmup_ratio`, um campo classico
        de `TrainingArguments`, foi silenciosamente descartado, e o treino rodou
        sem aquecimento da taxa de aprendizado.

        Consultamos entao TRES fontes, da mais confiavel para a menos:
          1. os campos da dataclass, que incluem os herdados;
          2. as anotacoes de tipo acumuladas na hierarquia;
          3. a assinatura de `__init__`, como ultimo recurso.
    """
    nomes: set[str] = set()

    if dataclasses.is_dataclass(classe):
        nomes |= {campo.name for campo in dataclasses.fields(classe)}

    for ancestral in getattr(classe, "__mro__", [classe]):
        nomes |= set(getattr(ancestral, "__annotations__", {}))

    with contextlib.suppress(TypeError, ValueError):
        nomes |= set(inspect.signature(classe.__init__).parameters)

    return nomes


def _filtrar_argumentos(classe, argumentos: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Mantem apenas os argumentos que a classe realmente aceita.

    Devolve tambem os descartados, para que o notebook os IMPRIMA. Um
    argumento silenciosamente ignorado e pior do que um erro: o treino roda
    com uma configuracao diferente da pretendida e ninguem percebe.
    """
    aceitos_pela_classe = _nomes_aceitos(classe)
    if not aceitos_pela_classe:
        # Nao conseguimos descobrir nada: melhor passar tudo e deixar a
        # biblioteca reclamar do que descartar em silencio.
        return argumentos, []

    aceitos = {k: v for k, v in argumentos.items() if k in aceitos_pela_classe}
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


def alinhar_precisao_dos_adaptadores(treinador) -> int:
    """
    Converte os parametros TREINAVEIS para float32 quando o treino usa fp16.

    O DEFEITO QUE ISTO CORRIGE:
        Os adaptadores LoRA nascem na precisao do modelo base. Varios modelos
        declaram `torch_dtype: bfloat16` no proprio config.json - o Qwen2.5 e um
        deles -, e o argumento que pediria float16 no carregamento mudou de nome
        entre versoes do transformers (`torch_dtype` virou `dtype`). Quando o
        nome nao bate, ele cai em **kwargs, e o modelo carrega em bf16
        silenciosamente.

        Numa GPU sem suporte a bfloat16, como a T4 do Colab gratuito, o treino
        roda em fp16 com GradScaler. O scaler nao sabe processar gradientes bf16,
        e o treino morre com:

            NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
            not implemented for 'BFloat16'

        A mensagem nao menciona dtype de adaptador, nao menciona o modelo, e
        aparece dentro do torch - a tres camadas de distancia da causa.

    POR QUE float32 E NAO float16:
        E a receita padrao do QLoRA: base congelada em 4 bits, adaptadores em
        precisao cheia, autocast cuidando da velocidade. Sao ~24 milhoes de
        parametros treinaveis; mante-los em fp32 custa cerca de 100 MB de VRAM,
        e evita tanto o problema do scaler quanto a instabilidade numerica de
        acumular gradientes em meia precisao.

    Returns:
        Quantos parametros foram convertidos. Zero significa que ja estavam
        corretos - o que e o caso quando o treino usa bf16 nativo.
    """
    import torch

    convertidos = 0
    for _, parametro in treinador.model.named_parameters():
        if parametro.requires_grad and parametro.dtype != torch.float32:
            parametro.data = parametro.data.to(torch.float32)
            convertidos += 1
    return convertidos


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

    treinador = SFTTrainer(**aceitos)

    # Alinha a precisao dos adaptadores ANTES de devolver o treinador. Feito
    # aqui, e nao no notebook, para que nenhuma execucao possa esquecer -
    # o sintoma so apareceria depois, dentro do torch, com uma mensagem que
    # nao aponta para a causa.
    if getattr(configuracao, "fp16", False):
        convertidos = alinhar_precisao_dos_adaptadores(treinador)
        if convertidos:
            print(
                f"{convertidos} parametro(s) treinavel(is) convertido(s) para float32 "
                f"(treino em fp16 exige gradientes float32 no GradScaler)"
            )

    return treinador


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
