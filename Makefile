# =============================================================================
# MedGraph - atalhos de execucao
# Tech Challenge Fase 3 - 8IADT
#
# Rode `make` ou `make ajuda` para ver todos os comandos disponiveis.
# Cada alvo abaixo corresponde a uma etapa do projeto descrita no README.
# =============================================================================

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
# Todos os alvos rodam com o repositorio no PYTHONPATH. Esse e o contrato do
# projeto: nao dependemos de `pip install -e .` ter funcionado, porque o modo
# editavel do setuptools se mostrou intermitente (ver caminhos.py).
RAIZ := $(shell pwd)
export PYTHONPATH := $(RAIZ):$(RAIZ)/src

# Repasse de flags aos scripts: `make avaliar -- --rapido`.
# O `make` nao encaminha argumentos para a receita; ele os interpreta como
# alvos. Colhemos aqui os que parecem flag e os devolvemos ao script, e a regra
# `--%` do fim do arquivo absorve o alvo falso que cada flag cria.
# Filtramos por `-%` em vez de "tudo que nao e o alvo atual" de proposito:
# assim `make dados avaliar` continua sendo dois alvos, e nao um alvo com o
# outro virando argumento.
ARGS = $(filter -%,$(MAKECMDGOALS))

.DEFAULT_GOAL := ajuda
.PHONY: ajuda setup ambiente testes lint formatar limpar limpar-logs \
        dados finetune-prep modelo avaliar indexar grafo diagrama app \
        rastreabilidade relatorio guia tudo

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
	$(PYTHON) scripts/verificar_ambiente.py $(ARGS)

# --- Qualidade ---------------------------------------------------------------
testes:  ## Roda a suite de testes automatizados
	$(PYTHON) -m pytest tests/ -v $(ARGS)

lint:  ## Verifica estilo e possiveis erros com o ruff
	$(PYTHON) -m ruff check src/ config/ tests/ scripts/ $(ARGS)

formatar:  ## Formata o codigo e ordena os imports
	$(PYTHON) -m ruff format src/ config/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ config/ tests/ scripts/

# --- Pipeline do projeto (na ordem das etapas) -------------------------------
dados:  ## Etapa 1 - baixa, anonimiza e cura os dados; gera o corpus hospitalar
	$(PYTHON) scripts/01_preparar_dados.py $(ARGS)

finetune-prep:  ## Etapa 2 - monta o dataset de fine-tuning para o Colab
	$(PYTHON) scripts/02_preparar_finetune.py $(ARGS)

modelo:  ## Etapa 3 - baixa o GGUF do HF Hub e registra o modelo no Ollama
	$(PYTHON) scripts/03_instalar_modelo.py $(ARGS)

avaliar:  ## Etapa 4 - compara base vs fine-tunado vs gpt-4o-mini e gera graficos
	$(PYTHON) scripts/04_avaliar.py $(ARGS)

indexar:  ## Etapa 5 - constroi o indice vetorial FAISS
	$(PYTHON) scripts/05_indexar.py $(ARGS)

grafo:  ## Etapa 7 - executa o fluxo LangGraph no terminal
	$(PYTHON) scripts/07_rodar_grafo.py $(ARGS)

diagrama:  ## Etapa 7 - gera os diagramas do grafo (ASCII, Mermaid e PNG)
	$(PYTHON) -m medgraph.grafo.diagrama $(ARGS)

app:  ## Etapa 8 - abre o painel visual em Streamlit
	.venv/bin/streamlit run src/medgraph/ui/app_streamlit.py

relatorio:  ## Gera docs/relatorio_tecnico.md com os numeros dos artefatos
	$(PYTHON) scripts/gerar_relatorio.py $(ARGS)

rastreabilidade:  ## Gera docs/rastreabilidade.md a partir das tags [REQ-xx]
	$(PYTHON) scripts/gerar_rastreabilidade.py $(ARGS)

# --- Limpeza -----------------------------------------------------------------
limpar:  ## Remove caches do Python e das ferramentas
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

limpar-logs:  ## Apaga logs e traces (cuidado: descarta a trilha de auditoria)
	rm -f logs/*.log logs/*.jsonl logs/auditoria/*.jsonl logs/traces/*.json
	@echo "Logs e traces removidos."

# --- Execucao completa -------------------------------------------------------
tudo: dados finetune-prep indexar avaliar diagrama rastreabilidade relatorio  ## Roda o pipeline inteiro
	@echo "Pipeline concluido. Abra o painel com: make app"

# -----------------------------------------------------------------------------
# Alvo falso para as flags repassadas via $(ARGS). Sem ele, `make avaliar --
# --rapido` roda a avaliacao e so entao aborta com "No rule to make target",
# dando a impressao de que o comando falhou quando ele ja executou.
# O padrao e restrito a `--%` de proposito: um alvo inexistente que nao comeca
# com hifen - `make teste` no lugar de `make testes` - continua falhando alto,
# em vez de virar um no-op silencioso.
--%:
	@:

guia:  ## Gera docs/MedGraph-Guia-do-Projeto.pdf a partir do Markdown
	$(PYTHON) scripts/gerar_guia_pdf.py $(ARGS)
