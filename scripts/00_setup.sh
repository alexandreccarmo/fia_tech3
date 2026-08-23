#!/usr/bin/env bash
# =============================================================================
# MedGraph - preparacao do ambiente de desenvolvimento
#
# O QUE ESTE SCRIPT FAZ:
#   1. Confere que existe um Python 3.12 na maquina;
#   2. Cria o ambiente virtual .venv;
#   3. Instala as dependencias de execucao e de desenvolvimento;
#   4. Instala o proprio projeto em modo editavel (pip install -e .);
#   5. Cria o .env a partir do .env.example, se ainda nao existir;
#   6. Roda a suite de testes para confirmar que a fundacao esta de pe.
#
# POR QUE PYTHON 3.12 E NAO O MAIS NOVO:
#   Parte do stack de ML (torch, faiss, sentence-transformers) ainda nao
#   publica wheels para 3.13/3.14. Fixar 3.12 evita a hora perdida
#   compilando dependencia da fonte.
#
# Uso:
#   bash scripts/00_setup.sh
# =============================================================================
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

VERDE='\033[0;32m'; AMARELO='\033[0;33m'; VERMELHO='\033[0;31m'; AZUL='\033[0;36m'; FIM='\033[0m'

titulo()  { echo -e "\n${AZUL}==> $1${FIM}"; }
ok()      { echo -e "${VERDE}  ok  ${FIM} $1"; }
aviso()   { echo -e "${AMARELO}  !   ${FIM} $1"; }
erro()    { echo -e "${VERMELHO}  x   ${FIM} $1"; }

echo "============================================================"
echo "  MedGraph - Tech Challenge Fase 3 (8IADT)"
echo "  Preparacao do ambiente"
echo "============================================================"

# -----------------------------------------------------------------------------
titulo "1/6  Procurando Python 3.12"
# -----------------------------------------------------------------------------
PY=""
for candidato in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
    if command -v "$candidato" >/dev/null 2>&1; then
        PY="$candidato"; break
    fi
done

if [ -z "$PY" ]; then
    erro "Python 3.12 nao encontrado."
    echo ""
    echo "  Instale com:"
    echo "    macOS   : brew install python@3.12"
    echo "    Ubuntu  : sudo apt install python3.12 python3.12-venv"
    exit 1
fi
ok "$PY ($("$PY" --version))"

# -----------------------------------------------------------------------------
titulo "2/6  Criando o ambiente virtual .venv"
# -----------------------------------------------------------------------------
if [ -d ".venv" ]; then
    aviso ".venv ja existe - reutilizando. Apague a pasta para recriar do zero."
else
    "$PY" -m venv .venv
    ok ".venv criado"
fi
./.venv/bin/python -m pip install --quiet --upgrade pip setuptools wheel
ok "pip atualizado"

# -----------------------------------------------------------------------------
titulo "3/6  Instalando dependencias (pode levar alguns minutos)"
# -----------------------------------------------------------------------------
aviso "O download do torch e do sentence-transformers e o passo mais demorado."
./.venv/bin/pip install --quiet -r requirements-dev.txt
ok "dependencias instaladas"

# -----------------------------------------------------------------------------
titulo "4/6  Instalando o projeto em modo editavel"
# -----------------------------------------------------------------------------
./.venv/bin/pip install --quiet -e .
ok "'import medgraph' e 'import config' disponiveis em qualquer diretorio"

# -----------------------------------------------------------------------------
titulo "5/6  Configurando o arquivo .env"
# -----------------------------------------------------------------------------
if [ -f ".env" ]; then
    aviso ".env ja existe - preservado (nao sobrescrevemos configuracao sua)."
else
    cp .env.example .env
    ok ".env criado a partir do .env.example"
    aviso "Edite o .env e preencha OPENAI_API_KEY se for usar a OpenAI."
fi

# -----------------------------------------------------------------------------
titulo "6/6  Rodando os testes da fundacao"
# -----------------------------------------------------------------------------
if ./.venv/bin/python -m pytest tests/ -q; then
    ok "todos os testes passaram"
else
    erro "algum teste falhou - veja a saida acima antes de prosseguir"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "  ${VERDE}Ambiente pronto.${FIM}"
echo "============================================================"
echo ""
echo "  Proximos passos:"
echo "    make ambiente    # diagnostico completo do ambiente"
echo "    make ajuda       # lista todos os comandos do projeto"
echo "    make dados       # Etapa 1: preparacao dos dados"
echo ""
