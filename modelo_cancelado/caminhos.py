"""
Resolucao dos caminhos de importacao do MedGraph.

O QUE FAZ:
    Coloca a raiz do repositorio e `src/` no `sys.path`, para que
    `import medgraph` e `import config` funcionem independentemente de onde o
    processo foi iniciado e de o projeto ter sido instalado com pip ou nao.

POR QUE EXISTE, EM VEZ DE SO USAR `pip install -e .`:
    O projeto usa layout `src/`, e o modo editavel do setuptools resolve isso
    instalando um arquivo `.pth` que registra um localizador dinamico em
    `sys.meta_path`. Durante o desenvolvimento deste projeto esse mecanismo se
    mostrou INTERMITENTE no ambiente de trabalho: o mesmo arquivo, com os
    mesmos bytes e as mesmas permissoes, ora era processado pelo modulo `site`
    na inicializacao do interpretador, ora nao - deixando o pacote invisivel
    sem emitir nenhum erro.

    Um projeto que sera clonado e executado por outra pessoa nao pode depender
    de um mecanismo que falha em silencio. Este modulo troca essa dependencia
    por tres linhas de codigo deterministicas, que funcionam em qualquer
    sistema operacional e sem instalacao previa.

    `pip install -e .` continua sendo feito pelo `make setup` e continua
    valendo quando funciona; este modulo apenas garante que o projeto rode
    tambem quando nao funciona.

COMO USAR:
    A primeira linha de qualquer script ou notebook do projeto:

        import caminhos  # noqa: F401  - resolve sys.path

    Os modulos internos de `medgraph` NAO precisam disto: uma vez que o
    processo entrou pelo ponto de entrada, os caminhos ja estao resolvidos.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parent
SRC: Path = RAIZ / "src"


def resolver() -> tuple[Path, Path]:
    """Insere a raiz e src/ no inicio do sys.path, sem duplicar."""
    for caminho in (SRC, RAIZ):
        texto = str(caminho)
        if texto not in sys.path:
            sys.path.insert(0, texto)
    return RAIZ, SRC


# Executa na importacao: o objetivo do modulo e ser um efeito colateral util.
resolver()
