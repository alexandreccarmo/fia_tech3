"""
Configuracao do pytest na raiz do repositorio.

POR QUE ESTE ARQUIVO EXISTE:
    O caminho recomendado para rodar o projeto e `pip install -e .`, que
    registra os pacotes `medgraph` e `config` no ambiente. Este conftest e
    uma rede de seguranca: garante que os testes rodem mesmo em um clone
    recem-baixado onde a instalacao editavel ainda nao foi feita - situacao
    provavel quando o professor abre o repositorio pela primeira vez.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

for caminho in (RAIZ, RAIZ / "src"):
    texto = str(caminho)
    if texto not in sys.path:
        sys.path.insert(0, texto)
