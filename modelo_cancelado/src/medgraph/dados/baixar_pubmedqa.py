"""
[REQ-1][REQ-E2] Download do PubMedQA.

O QUE FAZ:
    Baixa o dataset PubMedQA do Hugging Face Hub e o grava, sem nenhuma
    transformacao, em data/raw/. O tratamento vem depois, em curadoria.py.

POR QUE MANTER UMA COPIA BRUTA:
    Separar "baixar" de "tratar" e o que torna o pipeline auditavel. Com a
    copia intacta em disco, e possivel reexecutar a curadoria com criterios
    diferentes e comparar resultados, sem depender de a fonte remota continuar
    identica. Se so guardassemos o dado ja processado, nenhuma decisao de
    curadoria seria reproduzivel nem contestavel.

O QUE E O PUBMEDQA:
    Perguntas de pesquisa biomedica derivadas de titulos de artigos do PubMed,
    acompanhadas do abstract correspondente (o contexto) e de uma decisao
    yes/no/maybe dada por especialistas. Cada registro tem:

      pubid           identificador do artigo no PubMed
      question        a pergunta de pesquisa
      context         o abstract estruturado (secoes + rotulos + termos MeSH)
      long_answer     a conclusao do artigo, em texto corrido
      final_decision  yes | no | maybe

    Tres subconjuntos:
      pqa_labeled     1.000 registros anotados por especialistas
      pqa_artificial  ~211.000 com rotulos gerados automaticamente
      pqa_unlabeled   ~61.000 sem rotulo (nao usamos)

COMO ESTE PROJETO USA CADA UM:
    pqa_labeled     dividido em treino/validacao/teste. O teste NUNCA e usado
                    no fine-tuning: e a base da avaliacao da Etapa 4.
    pqa_artificial  amostra usada como volume de treino, para o modelo
                    aprender o formato da tarefa antes de ver os dados de
                    especialista.

Uso:
    python -m medgraph.dados.baixar_pubmedqa
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

REPO_HF = "qiaojin/PubMedQA"

# Subconjuntos que este projeto utiliza, com o nome do arquivo de saida.
# O formato de gravacao e JSON Lines, e nao um array JSON unico, por causa do
# tamanho: o subconjunto artificial tem 211 mil registros e ~735 MB. Carregar
# isso com json.load() consome varios GB de RAM de uma so vez. Com JSONL, a
# curadoria le linha a linha e trabalha com memoria constante.
SUBCONJUNTOS: dict[str, str] = {
    "pqa_labeled": "pubmedqa_labeled.jsonl",
    "pqa_artificial": "pubmedqa_artificial.jsonl",
}


def _normalizar_registro(registro: dict[str, Any]) -> dict[str, Any]:
    """
    Achata o registro do Hugging Face para um formato estavel em disco.

    O campo `context` vem como um dicionario com listas paralelas
    (`contexts` e `labels`). Guardamos as duas coisas: o texto ja unido, que
    e o que o modelo vai ler, e as secoes separadas, que preservam a
    estrutura do abstract caso alguma analise futura precise dela.
    """
    contexto = registro.get("context") or {}
    trechos: list[str] = list(contexto.get("contexts") or [])
    rotulos: list[str] = list(contexto.get("labels") or [])

    return {
        "pubid": registro.get("pubid"),
        "question": (registro.get("question") or "").strip(),
        "contexto_texto": "\n\n".join(t.strip() for t in trechos if t and t.strip()),
        "contexto_secoes": [
            {"rotulo": r, "texto": t}
            for r, t in zip(rotulos or [""] * len(trechos), trechos, strict=False)
        ],
        "mesh": list(contexto.get("meshes") or []),
        "long_answer": (registro.get("long_answer") or "").strip(),
        "final_decision": (registro.get("final_decision") or "").strip().lower(),
    }


def baixar_subconjunto(nome: str, cfg: Settings | None = None) -> list[dict[str, Any]]:
    """Baixa um subconjunto do PubMedQA e devolve os registros normalizados."""
    cfg = cfg or obter_settings()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "A biblioteca 'datasets' nao esta instalada. Rode: make setup"
        ) from exc

    with etapa(f"download:{nome}", tipo=TipoEvento.FIM_ETAPA, repositorio=REPO_HF):
        log.info("Baixando %s / %s (a primeira vez leva alguns minutos)...", REPO_HF, nome)
        dataset = load_dataset(REPO_HF, nome)

    # Todos os subconjuntos do PubMedQA vem em um unico split chamado 'train'.
    # A divisao em treino/validacao/teste e feita por nos, em curadoria.py,
    # com semente fixa - assim ela e reproduzivel e documentavel.
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    registros = [_normalizar_registro(r) for r in split]

    distribuicao = Counter(r["final_decision"] for r in registros)
    log.info(
        "%s: %d registros | distribuicao de rotulos: %s",
        nome,
        len(registros),
        dict(distribuicao),
    )
    registrar(
        TipoEvento.FIM_ETAPA,
        f"Subconjunto {nome} baixado",
        etapa=f"download:{nome}",
        conclusao=True,
        registros=len(registros),
        distribuicao=dict(distribuicao),
    )
    return registros


def baixar(cfg: Settings | None = None, *, forcar: bool = False) -> dict[str, int]:
    """
    Baixa os subconjuntos necessarios e grava em data/raw/.

    Args:
        forcar: rebaixa mesmo que o arquivo ja exista. Sem isso, executar o
            pipeline duas vezes nao repete um download de centenas de MB.

    Returns:
        Quantidade de registros por subconjunto.
    """
    cfg = cfg or obter_settings()
    cfg.dir_dados_brutos.mkdir(parents=True, exist_ok=True)
    resultado: dict[str, int] = {}

    for nome, arquivo in SUBCONJUNTOS.items():
        destino = cfg.dir_dados_brutos / arquivo

        if destino.exists() and not forcar:
            with destino.open(encoding="utf-8") as arquivo_existente:
                quantidade = sum(1 for _ in arquivo_existente)
            log.info("%s ja existe (%d registros) - pulando o download.", arquivo, quantidade)
            resultado[nome] = quantidade
            continue

        registros = baixar_subconjunto(nome, cfg)
        with destino.open("w", encoding="utf-8") as saida:
            for registro in registros:
                saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
        tamanho_mb = destino.stat().st_size / 1024**2
        log.info("Gravado %s (%d registros, %.1f MB)", destino.name, len(registros), tamanho_mb)
        resultado[nome] = len(registros)

    return resultado


def iterar(nome: str, cfg: Settings | None = None) -> Iterator[dict[str, Any]]:
    """
    Percorre um subconjunto ja baixado, um registro por vez.

    E a forma preferida de ler o subconjunto artificial: mantem a memoria
    constante independentemente do tamanho do arquivo.
    """
    cfg = cfg or obter_settings()
    caminho = cfg.dir_dados_brutos / SUBCONJUNTOS[nome]
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} nao existe. Rode primeiro: make dados")

    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                yield json.loads(linha)


def carregar(nome: str, cfg: Settings | None = None) -> list[dict[str, Any]]:
    """
    Le um subconjunto inteiro para a memoria.

    Use apenas com pqa_labeled (1.000 registros). Para o artificial, prefira
    `iterar()` - carregar 211 mil registros de uma vez consome varios GB.
    """
    return list(iterar(nome, cfg))


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Download do PubMedQA", subtitulo=REPO_HF)
    print(baixar())
