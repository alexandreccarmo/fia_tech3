# Matriz de rastreabilidade

> **Documento gerado automaticamente.** Não edite à mão — rode `make rastreabilidade`.
> Última geração: 2026-08-23

Este documento liga cada exigência do enunciado do Tech Challenge (Fase 3 — 8IADT)
ao ponto exato do código onde ela é atendida. As referências vêm das tags
`[REQ-xx]` escritas nas docstrings do projeto.

## Resumo da cobertura

| Requisitos no catálogo | 13 |
| --- | --- |
| **Com implementação identificada** | **9** |
| Ainda sem implementação | 4 |
| Total de referências no código | 80 |

### Requisitos ainda sem cobertura

São itens previstos para etapas seguintes do projeto.

| Código | Categoria | Descrição |
| --- | --- | --- |
| `REQ-E1` | Entregaveis | Codigo-fonte com os fluxos do LangGraph. |
| `REQ-E2` | Entregaveis | Dataset anonimizado ou exemplo de dados sinteticos. |
| `REQ-E3` | Entregaveis | Relatorio tecnico detalhado com explicacao do processo de fine-tuning, descricao do assistente medico criado, diagrama do fluxo LangChain e avaliacao do modelo com analise dos resultados. |
| `REQ-E4` | Entregaveis | Video de ate 15 minutos demonstrando o treinamento e funcionamento da LLM personalizada, a execucao de um fluxo automatizado, respostas a perguntas clinicas contextualizadas e os logs e a validacao das respostas. |

## Detalhamento por requisito

### Fine-tuning

#### ✅ `REQ-1` — Realizar o fine-tuning de um modelo LLM (como LLaMA, Falcon ou outro) utilizando protocolos medicos do hospital, exemplos de perguntas frequentes feitas por medicos e modelos de laudos, receitas e procedimentos internos.

*Origem no PDF: PDF pag. 2, item 1*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `src/medgraph/__init__.py` | 20 | finetune/         preparo do dataset e notebooks do Colab     [REQ-1] |
| `src/medgraph/dados/baixar_pubmedqa.py` | 2 | [REQ-1][REQ-E2] Download do PubMedQA. |
| `src/medgraph/finetune/__init__.py` | 1 | """[REQ-1] Preparo do dataset de fine-tuning e artefatos do treino no Colab.""" |
| `src/medgraph/finetune/colab_utils.py` | 2 | [REQ-1] Utilidades do fine-tuning executado no Google Colab. |
| `src/medgraph/finetune/preparar_dataset_sft.py` | 2 | [REQ-1] Montagem do dataset de fine-tuning supervisionado. |
| `tests/test_dados.py` | 352 | DATASET DE FINE-TUNING  [REQ-1] |
| `tests/test_dados.py` | 381 | """[REQ-1] Sem isso, "maybe" ficaria abaixo de 1% do dataset.""" |

#### ✅ `REQ-1a` — Preparar os dados com tecnicas de preprocessing, anonimizacao e curadoria.

*Origem no PDF: PDF pag. 2, item 1, segundo marcador*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/politicas.yaml` | 140 | mesmo que tenha entrado pelo prontuario. [REQ-1a] |
| `src/medgraph/__init__.py` | 19 | dados/            download, anonimizacao e curadoria          [REQ-1a] |
| `src/medgraph/auditoria.py` | 97 | ANONIMIZACAO = "anonimizacao"  # PII removida                      [REQ-1a] |
| `src/medgraph/auditoria.py` | 140 | Chamada uma vez no bootstrap, pelo modulo de anonimizacao. [REQ-1a] |
| `src/medgraph/dados/__init__.py` | 1 | """[REQ-1a][REQ-E2] Aquisicao, anonimizacao e curadoria dos dados.""" |
| `src/medgraph/dados/anonimizador.py` | 2 | [REQ-1a] Anonimizacao de dados pessoais e de saude. |
| `src/medgraph/dados/anonimizador.py` | 453 | [REQ-1a][REQ-3b] Faz a trilha de auditoria passar por este modulo. |
| `src/medgraph/dados/curadoria.py` | 2 | [REQ-1a] Curadoria e divisao do PubMedQA. |
| `tests/test_dados.py` | 27 | ANONIMIZAÇÃO  [REQ-1a] |
| `tests/test_dados.py` | 152 | CURADORIA  [REQ-1a] |
| `tests/test_fundacao.py` | 422 | """[REQ-1a] Nenhum dado pessoal chega ao disco pela trilha.""" |

### Assistente LangChain

#### ✅ `REQ-2` — Utilizar o LangChain para construir um pipeline que integre a LLM customizada.

*Origem no PDF: PDF pag. 3, item 2*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `src/medgraph/__init__.py` | 22 | llm/              provedores de modelo e controle de custo    [REQ-2] |
| `src/medgraph/__init__.py` | 25 | chains/           pipelines LangChain                         [REQ-2] |
| `src/medgraph/chains/__init__.py` | 1 | """[REQ-2] Pipelines LangChain: triagem, RAG e geracao de documentos.""" |
| `src/medgraph/chains/prompts.py` | 2 | [REQ-2][REQ-3a][REQ-3c] Prompts do MedGraph. |
| `src/medgraph/llm/__init__.py` | 1 | """[REQ-2] Provedores de modelo de linguagem e controle de consumo.""" |
| `src/medgraph/llm/provider.py` | 2 | [REQ-2] Provedor de modelo de linguagem. |
| `src/medgraph/rag/__init__.py` | 1 | """[REQ-2][REQ-3c] Indexacao vetorial e recuperacao de evidencia com fontes.""" |

#### ✅ `REQ-2a` — Realizar consultas em base de dados estruturadas (como prontuarios e registros).

*Origem no PDF: PDF pag. 3, item 2*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/settings.py` | 227 | """Base SQLite com os prontuarios estruturados. [REQ-2a]""" |
| `scripts/demo_fundacao.py` | 88 | """Simula a leitura da base estruturada de pacientes. [REQ-2a]""" |
| `src/medgraph/__init__.py` | 24 | prontuario/       acesso a base estruturada de pacientes      [REQ-2a] |
| `src/medgraph/auditoria.py` | 100 | BANCO = "banco"                # consulta ao prontuario            [REQ-2a] |
| `src/medgraph/dados/construir_banco.py` | 2 | [REQ-2a][REQ-E2] Construcao da base estruturada de prontuarios. |
| `src/medgraph/prontuario/__init__.py` | 1 | """[REQ-2a][REQ-2b] Acesso a base estruturada de prontuarios do hospital.""" |
| `tests/test_dados.py` | 266 | BASE DE PRONTUÁRIOS  [REQ-2a] |

#### ✅ `REQ-2b` — Contextualizar as respostas da LLM com informacoes atualizadas do paciente.

*Origem no PDF: PDF pag. 3, item 2*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `src/medgraph/prontuario/__init__.py` | 1 | """[REQ-2a][REQ-2b] Acesso a base estruturada de prontuarios do hospital.""" |

### Seguranca e validacao

#### ✅ `REQ-3a` — Definir limites de atuacao do assistente para evitar sugestoes improprias (ex.: nunca prescrever diretamente, sem validacao humana).

*Origem no PDF: PDF pag. 3, item 3 (trecho destacado no enunciado)*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/politicas.yaml` | 2 | [REQ-3a] POLITICAS DE ATUACAO DO ASSISTENTE MEDGRAPH |
| `config/politicas.yaml` | 128 | [REQ-3a] Nenhuma posologia sai sem marcacao explicita de revisao humana. |
| `config/settings.py` | 116 | 4) COMPORTAMENTO DO GRAFO  [REQ-3a] |
| `config/settings.py` | 284 | """Regras declarativas de guardrail. [REQ-3a]""" |
| `scripts/03_instalar_modelo.py` | 101 | PROMPT DE SISTEMA — limites de atuação do assistente  [REQ-3a] |
| `scripts/demo_fundacao.py` | 75 | """Simula a verificacao de escopo e a anonimizacao da pergunta. [REQ-3a]""" |
| `scripts/demo_fundacao.py` | 128 | """Simula a validacao da resposta antes da entrega. [REQ-3a][REQ-3c]""" |
| `src/medgraph/__init__.py` | 26 | guardrails/       limites de atuacao do assistente            [REQ-3a] |
| `src/medgraph/auditoria.py` | 96 | GUARDRAIL = "guardrail"        # aprovacao/reprovacao de politica  [REQ-3a] |
| `src/medgraph/auditoria.py` | 103 | VALIDACAO_HUMANA = "validacao_humana"  # human-in-the-loop         [REQ-3a] |
| `src/medgraph/chains/prompts.py` | 2 | [REQ-2][REQ-3a][REQ-3c] Prompts do MedGraph. |
| `src/medgraph/guardrails/__init__.py` | 1 | """[REQ-3a] Limites de atuacao: guardrails de entrada, de saida e regras clinicas.""" |
| `tests/test_dados.py` | 356 | """[REQ-3a] O limite mais importante precisa estar no prompt.""" |
| `tests/test_fundacao.py` | 521 | POLITICAS  [REQ-3a] |
| `tests/test_fundacao.py` | 557 | """[REQ-3a] O item mais sensivel do enunciado, verificado por teste.""" |

#### ✅ `REQ-3b` — Implementar logging detalhado para rastreamento e auditoria.

*Origem no PDF: PDF pag. 3, item 3*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/settings.py` | 95 | 2) CONTROLE DE CUSTO  [REQ-3b] |
| `config/settings.py` | 122 | 5) LOGGING E AUDITORIA  [REQ-3b] |
| `config/settings.py` | 238 | """Trilha de auditoria em JSONL, um arquivo por dia. [REQ-3b]""" |
| `config/settings.py` | 244 | """Trace completo de cada consulta, um JSON por trace_id. [REQ-3b]""" |
| `config/settings.py` | 330 | disco. [REQ-3b] |
| `src/medgraph/__init__.py` | 17 | logging_config.py logging em tres destinos                    [REQ-3b] |
| `src/medgraph/__init__.py` | 18 | auditoria.py      trilha de auditoria e trace por consulta    [REQ-3b] |
| `src/medgraph/auditoria.py` | 2 | [REQ-3b] Trilha de auditoria do MedGraph. |
| `src/medgraph/auditoria.py` | 513 | [REQ-3b] Decorator que audita automaticamente um no do grafo. |
| `src/medgraph/dados/anonimizador.py` | 435 | """Publica o resumo da anonimizacao na trilha de auditoria. [REQ-3b]""" |
| `src/medgraph/dados/anonimizador.py` | 453 | [REQ-1a][REQ-3b] Faz a trilha de auditoria passar por este modulo. |
| `src/medgraph/llm/custo.py` | 2 | [REQ-3b] Contabilidade de consumo e trava de orcamento. |
| `src/medgraph/llm/provider.py` | 78 | CALLBACK DE CUSTO  [REQ-3b] |
| `src/medgraph/logging_config.py` | 2 | [REQ-3b] Configuracao de logging do MedGraph. |
| `src/medgraph/requisitos.py` | 23 | Uma docstring que comeca com "[REQ-3b][REQ-3c]" significa que aquele |
| `tests/test_fundacao.py` | 84 | """[REQ-3b] A configuracao vai para a auditoria; o segredo, nunca.""" |
| `tests/test_fundacao.py` | 143 | LOGGING  [REQ-3b] |
| `tests/test_fundacao.py` | 239 | TRILHA DE AUDITORIA  [REQ-3b] |
| `tests/test_fundacao.py` | 434 | CONTROLE DE CUSTO  [REQ-3b] |
| `tests/test_fundacao.py` | 493 | """[REQ-3b] Consumo tambem e informacao auditavel.""" |

#### ✅ `REQ-3c` — Garantir explainability das respostas da LLM (exemplo: indicar a fonte da informacao utilizada na resposta).

*Origem no PDF: PDF pag. 3, item 3*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/politicas.yaml` | 120 | [REQ-3c] Explainability: toda afirmacao clinica precisa apontar de onde veio. |
| `scripts/demo_fundacao.py` | 98 | """Simula a busca no indice vetorial, devolvendo as fontes. [REQ-3c]""" |
| `scripts/demo_fundacao.py` | 128 | """Simula a validacao da resposta antes da entrega. [REQ-3a][REQ-3c]""" |
| `src/medgraph/__init__.py` | 23 | rag/              indice vetorial e recuperacao com fontes    [REQ-3c] |
| `src/medgraph/auditoria.py` | 99 | RECUPERACAO = "recuperacao"    # busca no indice vetorial          [REQ-3c] |
| `src/medgraph/chains/prompts.py` | 2 | [REQ-2][REQ-3a][REQ-3c] Prompts do MedGraph. |
| `src/medgraph/rag/__init__.py` | 1 | """[REQ-2][REQ-3c] Indexacao vetorial e recuperacao de evidencia com fontes.""" |
| `src/medgraph/requisitos.py` | 23 | Uma docstring que comeca com "[REQ-3b][REQ-3c]" significa que aquele |
| `tests/test_fundacao.py` | 353 | requisito e explainability por citacao, isso e inaceitavel. [REQ-3c] |
| `tests/test_fundacao.py` | 399 | que se quer poder auditar depois, item por item. [REQ-3c] |

### Organizacao do codigo

#### ✅ `REQ-4` — Projeto modularizado em Python com instrucoes completas no README.

*Origem no PDF: PDF pag. 3, item 4*

| Arquivo | Linha | Contexto |
| --- | ---: | --- |
| `config/settings.py` | 2 | [REQ-4] Configuracao centralizada do MedGraph. |
| `scripts/gerar_rastreabilidade.py` | 3 | [REQ-4] Gerador da matriz de rastreabilidade requisito -> codigo. |

### Entregaveis

#### ⏳ `REQ-E1` — Codigo-fonte com os fluxos do LangGraph.

*Origem no PDF: PDF pag. 3, Entregaveis / Repositorio Git*

_Sem implementação identificada até o momento._

#### ⏳ `REQ-E2` — Dataset anonimizado ou exemplo de dados sinteticos.

*Origem no PDF: PDF pag. 3, Entregaveis / Repositorio Git*

_Sem implementação identificada até o momento._

#### ⏳ `REQ-E3` — Relatorio tecnico detalhado com explicacao do processo de fine-tuning, descricao do assistente medico criado, diagrama do fluxo LangChain e avaliacao do modelo com analise dos resultados.

*Origem no PDF: PDF pag. 3-4, Entregaveis*

_Sem implementação identificada até o momento._

#### ⏳ `REQ-E4` — Video de ate 15 minutos demonstrando o treinamento e funcionamento da LLM personalizada, a execucao de um fluxo automatizado, respostas a perguntas clinicas contextualizadas e os logs e a validacao das respostas.

*Origem no PDF: PDF pag. 4*

_Sem implementação identificada até o momento._

