"""
[REQ-1] Monta o adapter final a partir de um checkpoint de treino.

POR QUE ISTO EXISTE:
    As celulas finais do notebook do Colab (curva de perda, cartao de treino,
    empacotamento) rodam DEPOIS do treino, e dependem de a sessao continuar de
    pe. Numa rodada de horas isso nao e garantido: basta o navegador cair, a
    energia faltar ou a VM ser reciclada para o treino terminar sem que ninguem
    consiga executar o que vem depois.

    Perder o resultado nesse ponto seria absurdo, porque ele ja existe. Todo
    checkpoint do treino contem os pesos do adapter, e o `trainer_state.json`
    guarda o historico completo de perdas. As celulas finais apenas copiam e
    formatam o que o checkpoint ja tem.

    Este script faz o mesmo trabalho na maquina local, sem GPU.

O QUE ELE NAO INVENTA:
    O cartao de treino gerado aqui traz apenas o que o checkpoint comprova. Os
    campos que so a sessao do Colab conhecia - modelo da GPU, versoes de
    biblioteca, duracao de relogio - ficam registrados como ausentes, e nao
    preenchidos por suposicao. Um cartao de procedencia que adivinha e pior do
    que um cartao incompleto.

Rodar com:
    make recuperar-adapter -- --checkpoint ~/Downloads/checkpoint-125
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from medgraph.finetune import colab_utils  # noqa: E402

DESTINO_PADRAO = Path("models/adapters/medgraph-llama32-3b-lora")

# Arquivos que compoem um adapter utilizavel. Os dois primeiros sao o adapter
# em si; o resto e o tokenizador, sem o qual o notebook de exportacao nao
# consegue fundir.
ESSENCIAIS = ("adapter_model.safetensors", "adapter_config.json")
ACOMPANHAM = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def localizar_checkpoint(indicado: Path) -> Path:
    """Aceita tanto o checkpoint quanto o diretorio que contem varios."""
    if (indicado / "adapter_config.json").exists():
        return indicado

    candidatos = sorted(
        (p for p in indicado.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    if candidatos:
        return candidatos[-1]

    raise SystemExit(
        f"nenhum adapter encontrado em {indicado}\n"
        "Aponte para a pasta do checkpoint (ex.: checkpoint-125) ou para o "
        "diretorio que a contem."
    )


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "--checkpoint", required=True, type=Path,
        help="pasta do checkpoint baixado do Drive, ou a que contem os checkpoints",
    )
    analisador.add_argument(
        "--destino", type=Path, default=DESTINO_PADRAO,
        help=f"onde gravar o adapter (padrao: {DESTINO_PADRAO})",
    )
    analisador.add_argument(
        "--modelo-base", default=None,
        help="identificador do modelo base usado no treino, para o cartao",
    )
    args = analisador.parse_args()

    origem = localizar_checkpoint(args.checkpoint.expanduser())
    print(f"checkpoint: {origem}")

    faltando = [nome for nome in ESSENCIAIS if not (origem / nome).exists()]
    if faltando:
        raise SystemExit(f"o checkpoint nao tem {faltando} - nao ha adapter aqui")

    destino = args.destino
    destino.mkdir(parents=True, exist_ok=True)

    copiados = []
    for nome in ESSENCIAIS + ACOMPANHAM:
        if (origem / nome).exists():
            shutil.copy2(origem / nome, destino / nome)
            copiados.append(nome)
    print(f"copiados {len(copiados)} arquivo(s):")
    for nome in copiados:
        print(f"  {nome}")

    estado_json = origem / "trainer_state.json"
    if not estado_json.exists():
        print(
            "\nAVISO: sem trainer_state.json. O adapter esta completo, mas nao "
            "ha historico de perdas para a curva nem para o cartao de treino."
        )
        return 0

    estado = json.loads(estado_json.read_text(encoding="utf-8"))
    historico = estado.get("log_history", [])

    grafico = colab_utils.grafico_de_perda(historico, destino / "curva_de_perda.png")
    print(f"\ncurva de perda: {grafico}")

    modelo_base = args.modelo_base or "(nao registrado no checkpoint)"
    colab_utils.salvar_metadados(
        destino / "cartao_de_treino.json",
        modelo_base=modelo_base,
        configuracao=colab_utils.CONFIG_PADRAO,
        # Estes tres so existiam na sessao do Colab. Declarar a ausencia e mais
        # honesto do que preencher com o que costuma ser verdade.
        versoes={"origem": "recuperado de checkpoint; versoes nao registradas"},
        gpu={"origem": "recuperado de checkpoint; GPU nao registrada"},
        historico=historico,
        exemplos_treino=0,
        exemplos_validacao=0,
        duracao_s=0.0,
    )
    print(f"cartao de treino: {destino / 'cartao_de_treino.json'}")
    print(f"\npassos treinados: {estado.get('global_step', '?')}")
    if modelo_base.startswith("("):
        print(
            "\nPasse --modelo-base para registrar qual modelo foi treinado. "
            "O notebook de exportacao le esse campo, e fundir na arquitetura "
            "errada nao falha: so responde pior."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
