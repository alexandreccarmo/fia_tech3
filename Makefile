# =============================================================================
# MedGraph - atalhos de execucao
# Tech Challenge Fase 3 - 8IADT
#
# Rode `make` ou `make ajuda` para ver todos os comandos disponiveis.
# Cada alvo abaixo corresponde a uma etapa do projeto descrita no README.
# =============================================================================

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
# Garante que `medgraph` e `config` sejam importaveis mesmo sem `pip install -e .`
export PYTHONPATH := .:src

.DEFAULT_GOAL := ajuda
.PHONY: ajuda setup ambiente testes lint formatar limpar limpar-logs \
        dados finetune-prep modelo avaliar indexar grafo diagrama app \
        rastreabilidade tudo

# -----------------------------------------------------------------------------
ajuda:  ## Lista os comandos disponiveis
	@echo ""
	@echo "  MedGraph - Tech Challenge Fase 3"
	@echo "  --------------------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- Ambiente ----------------------------------------------------------------
setup:  ## Cria o venv (Python 3.12) e instala todas as dependencias
	bash scripts/00_setup.sh

ambiente:  ## Verifica se o ambiente esta pronto para rodar o projeto
	$(PYTHON) scripts/verificar_ambiente.py

# --- Qualidade ---------------------------------------------------------------
testes:  ## Roda a suite de testes automatizados
	$(PYTHON) -m pytest tests/ -v

lint:  ## Verifica estilo e possiveis erros com o ruff
	$(PYTHON) -m ruff check src/ config/ tests/ scripts/

formatar:  ## Formata o codigo e ordena os imports
	$(PYTHON) -m ruff format src/ config/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ config/ tests/ scripts/

# --- Pipeline do projeto (na ordem das etapas) -------------------------------
dados:  ## Etapa 1 - baixa, anonimiza e cura os dados; gera o corpus hospitalar
	$(PYTHON) scripts/01_preparar_dados.py

finetune-prep:  ## Etapa 2 - monta o dataset de fine-tuning para o Colab
	$(PYTHON) scripts/02_preparar_finetune.py

modelo:  ## Etapa 3 - baixa o GGUF do HF Hub e registra o modelo no Ollama
	$(PYTHON) scripts/03_instalar_modelo.py

avaliar:  ## Etapa 4 - compara base vs fine-tunado vs gpt-4o-mini e gera graficos
	$(PYTHON) scripts/04_avaliar.py

indexar:  ## Etapa 5 - constroi o indice vetorial FAISS
	$(PYTHON) scripts/05_indexar.py

grafo:  ## Etapa 7 - executa o fluxo LangGraph no terminal
	$(PYTHON) scripts/07_rodar_grafo.py

diagrama:  ## Etapa 7 - gera os diagramas do grafo (ASCII, Mermaid e PNG)
	$(PYTHON) -m medgraph.grafo.diagrama

app:  ## Etapa 8 - abre o painel visual em Streamlit
	.venv/bin/streamlit run src/medgraph/ui/app_streamlit.py

rastreabilidade:  ## Gera docs/rastreabilidade.md a partir das tags [REQ-xx]
	$(PYTHON) scripts/gerar_rastreabilidade.py

# --- Limpeza -----------------------------------------------------------------
limpar:  ## Remove caches do Python e das ferramentas
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

limpar-logs:  ## Apaga logs e traces (cuidado: descarta a trilha de auditoria)
	rm -f logs/*.log logs/*.jsonl logs/auditoria/*.jsonl logs/traces/*.json
	@echo "Logs e traces removidos."

# --- Execucao completa -------------------------------------------------------
tudo: dados finetune-prep indexar avaliar diagrama  ## Roda o pipeline inteiro
	@echo "Pipeline concluido. Abra o painel com: make app"
