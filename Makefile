# =============================================================================
# MedGraph Lite - atalhos
# O projeto roda inteiro no Colab; estes alvos servem para conferir o codigo
# na maquina, sem GPU.
# =============================================================================

PYTHON := .venv/bin/python
export PYTHONPATH := $(shell pwd)

.DEFAULT_GOAL := ajuda
.PHONY: ajuda setup testes lint formatar demo dataset limpar

ajuda:  ## Lista os comandos
	@echo ""
	@echo "  MedGraph Lite - Tech Challenge Fase 3"
	@echo "  ----------------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  O notebook roda no Colab:"
	@echo "  https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb"
	@echo ""

# Qual python3 cria o venv. O projeto usa sintaxe de 3.10+ (`X | None`,
# `zip(strict=)`), e o macOS ainda traz um 3.9 em /usr/bin: sem esta checagem o
# `make setup` termina com sucesso e os testes falham depois com SyntaxError,
# num arquivo que nao tem nada de errado.
PY_SETUP ?= python3

setup:  ## Cria o venv e instala as dependencias
	@$(PY_SETUP) -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
		|| { echo ""; \
		     echo "  Este projeto precisa de Python 3.10 ou mais novo."; \
		     echo "  O '$(PY_SETUP)' desta maquina e:"; \
		     $(PY_SETUP) -V; \
		     echo ""; \
		     echo "  Aponte para outro:  make setup PY_SETUP=/opt/homebrew/bin/python3"; \
		     echo ""; exit 1; }
	$(PY_SETUP) -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q -r requirements.txt pytest ruff
	@echo "Pronto. Rode: make testes"

testes:  ## Roda a suite (nao precisa de GPU)
	$(PYTHON) -m pytest tests/ -v

lint:  ## Verifica estilo
	$(PYTHON) -m ruff check medgraph_lite/ tests/

formatar:  ## Formata e ordena imports
	$(PYTHON) -m ruff format medgraph_lite/ tests/
	$(PYTHON) -m ruff check --fix medgraph_lite/ tests/

demo:  ## Demonstra o grafo no terminal, com modelo simulado
	$(PYTHON) demo.py

dataset:  ## Reexporta o dataset sintetico para data/*.jsonl
	$(PYTHON) -c "from medgraph_lite import dados; \
	              print(dados.exportar_dataset('data'))"

limpar:  ## Remove caches e artefatos gerados
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.png *.db *.jsonl *.log
