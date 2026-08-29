# MedGraph — Assistente Clínico Auditável

**Tech Challenge — Fase 3 · Pós-Tech 8IADT**

Assistente virtual médico construído sobre uma LLM ajustada por *fine-tuning*, orquestrado
com **LangChain** e **LangGraph**, com trilha de auditoria completa, limites de atuação
explícitos e rastreabilidade de fontes em toda resposta.

---

## Integrantes do grupo

| Nome | RM |
| --- | --- |
| Alexandre Carneiro do Carmo | 370980 |
| Brunno Costa Castigrini | 371429 |
| Pedro Henrique Azevedo Aragão | 373481 |
| Valter Willian de Oliveira Filho | 370979 |

---

## 1. Propósito

O enunciado da Fase 3 pede um assistente virtual médico treinado com dados próprios de um
hospital, capaz de auxiliar em condutas clínicas, responder dúvidas do corpo médico e
sugerir procedimentos com base em protocolos internos — coordenando, além disso, **fluxos
de decisão automatizados e seguros**.

O **MedGraph** atende a isso no cenário fictício do **Hospital Vida Plena**. Diante de uma
pergunta clínica, o sistema:

1. **Filtra a entrada** — remove dados identificáveis e recusa pedidos fora de escopo;
2. **Classifica a intenção** — dúvida clínica, consulta ao paciente, exames pendentes,
   conduta terapêutica ou resumo de prontuário;
3. **Consulta o prontuário estruturado** — alergias, medicações em uso, exames pendentes;
4. **Recupera evidência** — busca semântica sobre artigos científicos e protocolos internos;
5. **Raciocina com a LLM ajustada** — resposta ancorada exclusivamente no contexto recuperado;
6. **Aplica regras clínicas** — alergias, interações medicamentosas, valores laboratoriais críticos;
7. **Valida a saída** — exige citação de fonte, bloqueia posologia sem revisão, insere o *disclaimer*;
8. **Escalona para validação humana** quando o risco ultrapassa o limiar configurado;
9. **Registra tudo** numa trilha de auditoria consultável.

### O princípio que rege o projeto

> **O assistente nunca prescreve. Ele apresenta evidência, aponta a fonte de cada afirmação
> e devolve a decisão ao médico responsável.**

Isso não é uma escolha estética: é o requisito 3 do enunciado, implementado em código
verificável — nos guardrails, no nó de validação humana e nos testes automatizados.

---

## 2. Status das etapas

O projeto é entregue em nove etapas, cada uma correspondendo a um commit no repositório.

| # | Etapa | Status |
| --- | --- | --- |
| 0 | **Fundação** — estrutura, configuração, logging, auditoria, controle de custo | ✅ |
| 1 | **Dados** — PubMedQA, anonimização, curadoria, corpus hospitalar sintético | ✅ |
| 2 | **Fine-tuning** — QLoRA sobre Llama-3.2-3B-Instruct no Google Colab | 📓 notebook pronto |
| 3 | **Export + Serve** — GGUF, Hugging Face Hub, Ollama | ✅ (modelo base servido) |
| 4 | **Avaliação** — comparativo de quatro sistemas, métricas e gráficos | ✅ |
| 5 | **RAG + Chains** — índice FAISS e pipelines LangChain com citação | ✅ |
| 6 | **Prontuário + Regras** — base SQLite e regras clínicas de segurança | ✅ |
| 7 | **LangGraph** — fluxo de 14 nós com guardrails e validação humana | ✅ |
| 8 | **UI + Documentação** — painel Streamlit, relatório técnico, roteiro do vídeo | ✅ |

### Sobre a Etapa 2

O fine-tuning é a **única** etapa que não roda no MacBook: exige GPU com CUDA. Os
notebooks estão prontos e documentados em `notebooks/colab/`, e o dataset de treino
está versionado no repositório — treinar é um `git clone` no Colab.

📘 **[Guia passo a passo do Colab](docs/guia_colab.md)** — pré-requisitos, link
direto para abrir os notebooks, o que esperar em cada célula e o que fazer quando
algo falha.

Enquanto o adapter não é produzido, o projeto roda de ponta a ponta com o **modelo
base servido sob a mesma persona e os mesmos parâmetros**, registrado no Ollama como
`medgraph-base`. Não é um atalho: é a coluna de referência que o comparativo da
Etapa 4 precisa de qualquer forma, servida em condições idênticas às do modelo
ajustado. Quando o GGUF ficar pronto, `make modelo -- --ajustado` o registra e a mesma
tabela de avaliação passa a trazer as duas colunas.

---

## 3. Arquitetura

```
                            ┌──────────────────────────┐
                            │   Médico (corpo clínico) │
                            └────────────┬─────────────┘
                                         │  pergunta + paciente_id
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                    LangGraph — fluxo de decisão                     │
   │                                                                     │
   │   guardrail_entrada → classificar_intenção → consultar_prontuário   │
   │        → recuperar_evidência → raciocínio_clínico → regras_clínicas │
   │        → guardrail_saída ⇄ reescrever → triagem_risco               │
   │        → validação_humana → emitir_alertas → montar_resposta        │
   └──────┬──────────────────┬───────────────────┬──────────────────┬────┘
          │                  │                   │                  │
          ▼                  ▼                   ▼                  ▼
   ┌────────────┐    ┌──────────────┐    ┌───────────────┐   ┌────────────┐
   │  SQLite    │    │ FAISS (RAG)  │    │ LLM ajustada  │   │  Auditoria │
   │ prontuários│    │ PubMedQA +   │    │ Llama-3.2-3B  │   │  JSONL +   │
   │  exames    │    │ protocolos   │    │ QLoRA/Ollama  │   │  traces    │
   └────────────┘    └──────────────┘    └───────────────┘   └────────────┘
```

### Decisões técnicas e a razão de cada uma

| Decisão | Por quê |
| --- | --- |
| **Fine-tuning no Google Colab** | QLoRA em GPU T4 gratuita treina um modelo de 3B em ~1 h; no Apple Silicon levaria horas |
| **Modelo servido pelo Ollama** | Mesmo padrão visto na Aula 05, roda offline, ~35 tok/s num M1, custo zero por consulta |
| **Embeddings locais** | `multilingual-e5-small` cobre inglês (artigos) e português (protocolos) sem custo recorrente |
| **FAISS** | Mesmo *vector store* usado nas aulas de LangGraph, persistido em disco |
| **Políticas em YAML** | Governança clínica não deveria exigir leitura de Python; mudança de política vira *diff* auditável |
| **Auditoria por `contextvars`** | Os nós do grafo mantêm assinatura limpa `(estado) -> estado`, sem carregar o `trace_id` como parâmetro |

---

## 4. Como rodar

### Pré-requisitos

| Item | Versão | Observação |
| --- | --- | --- |
| Python | **3.12** | O stack de ML ainda não tem *wheels* para 3.13/3.14 |
| Ollama | qualquer | `brew install ollama` — necessário a partir da Etapa 3 |
| Espaço em disco | ~12 GB | torch, modelo base, GGUF e datasets |
| Conta OpenAI | opcional | Apenas para o comparativo da Etapa 4 (custo total < US$ 1) |

### Instalação

```bash
git clone https://github.com/alexandreccarmo/fia_tech3.git
cd fia_tech3
make setup
```

O `make setup` cria o ambiente virtual, instala as dependências, instala o projeto em modo
editável, gera o `.env` a partir do `.env.example` e roda os testes.

### Verificar se está tudo certo

```bash
make ambiente
```

Apresenta uma tabela item a item — versão do Python, pacotes, `.env`, diretórios, Ollama,
artefatos de modelo e espaço em disco. Itens marcados como **AVISO** referem-se a artefatos
que só passam a existir em etapas posteriores; **FALHA** indica algo que impede a execução.

### Configurar a chave da OpenAI (opcional)

Edite o `.env` e preencha:

```bash
OPENAI_API_KEY=sk-proj-...
```

Recomendamos criar a chave dentro de um **projeto dedicado** na plataforma da OpenAI, com
limite de gasto definido. O projeto inteiro consome menos de US$ 1, e o próprio código
impõe um teto por sessão através de `MAX_CUSTO_USD_SESSAO`.

### Ver a fundação funcionando

```bash
.venv/bin/python scripts/demo_fundacao.py
```

Simula uma consulta clínica passando por cinco nós instrumentados e mostra, ao vivo, o log
colorido, a linha do tempo com latência por etapa, a trilha JSONL, o arquivo de *trace*, a
contabilidade de tokens e a trava de orçamento em ação.

---

## 5. Comandos disponíveis

```bash
make ajuda
```

| Comando | O que faz |
| --- | --- |
| `make setup` | Cria o venv e instala tudo |
| `make ambiente` | Diagnóstico visual do ambiente |
| `make testes` | Roda a suíte de testes |
| `make lint` / `make formatar` | Verifica e corrige estilo |
| `make dados` | **Etapa 1** — baixa, anonimiza, cura e monta todos os datasets |
| `make finetune-prep` | **Etapa 2** — remonta o dataset de fine-tuning |
| `make modelo` | **Etapa 3** — registra o modelo no Ollama |
| `make avaliar` | **Etapa 4** — avaliação comparativa e gráficos |
| `make indexar` | **Etapa 5** — constrói o índice FAISS |
| `make grafo` | **Etapa 7** — roteiro de demonstração no terminal |
| `make diagrama` | **Etapa 7** — gera os diagramas do grafo |
| `make app` | **Etapa 8** — abre o painel Streamlit |
| `make relatorio` | Regenera `docs/relatorio_tecnico.md` com os números atuais |
| `make rastreabilidade` | Regenera `docs/rastreabilidade.md` |
| `make tudo` | Pipeline completo, do download aos documentos |
| `make limpar-logs` | Apaga logs e *traces* |

### Demonstrações rápidas

```bash
make grafo                          # 4 casos, cada um por um caminho do grafo
make grafo -- --diagrama            # só desenha o fluxo
make grafo -- --pendentes           # fila de validação médica
make avaliar -- --rapido            # avaliação em 30 casos (~3 min)
.venv/bin/python scripts/demo_fundacao.py   # infraestrutura de auditoria
```

---

## 5.1 O que o assistente faz, em um exemplo

Pergunta de um médico, com o paciente `PAC-0001` vinculado:

> *"Qual a conduta antibiótica inicial para sepse de foco pulmonar neste paciente?"*

O que acontece:

1. **guardrail de entrada** — a pergunta é limpa de identificadores e aprovada
2. **classificar intenção** — `conduta_terapeutica`, a de maior risco
3. **consultar prontuário** — 72 anos, UTI, **alergia grave a penicilina
   (anafilaxia)**, lactato 4,5 mmol/L crítico, creatinina alterada
4. **recuperar evidência** — trechos de protocolo interno e literatura, marcados
   `[P1]`, `[P2]`, `[E1]`
5. **raciocínio clínico** — a LLM responde citando as fontes
6. **regras clínicas** — a resposta sugere uma cefalosporina; a tabela
   farmacológica identifica que **ceftriaxona é betalactâmico, a mesma classe da
   penicilina** → conflito **CRÍTICO**
7. **guardrail de saída** — citação presente, fontes existem, posologia marcada
8. **triagem de risco** — escore 1,0, acima do limiar de 0,6
9. **emitir alertas** — 3 alertas, o primeiro deles o conflito de alergia
10. **a execução PARA** — aguardando validação de um médico responsável

O médico validador vê os alertas, registra o parecer, e só então o fluxo retoma e
a resposta é liberada.

---

---

## 6. Estrutura do projeto

```
fia_tech3/
├── config/
│   ├── settings.py              Configuração central (lê o .env, valida, deriva caminhos)
│   └── politicas.yaml           Limites de atuação do assistente, em formato declarativo
│
├── src/medgraph/
│   ├── requisitos.py            Catálogo dos requisitos do enunciado (tags [REQ-xx])
│   ├── logging_config.py        Logging em três destinos
│   ├── auditoria.py             Trilha de auditoria, trace por consulta, @instrumentar
│   ├── dados/                   Download, anonimização, curadoria, corpus sintético
│   ├── finetune/                Preparo do dataset e artefatos do Colab
│   ├── avaliacao/               Métricas, comparativos e gráficos
│   ├── llm/                     Provedores de modelo e controle de custo
│   ├── rag/                     Indexação vetorial e recuperação com fontes
│   ├── prontuario/              Acesso à base estruturada de pacientes
│   ├── chains/                  Pipelines LangChain
│   ├── guardrails/              Guardrails de entrada e saída, regras clínicas
│   ├── grafo/                   Fluxo LangGraph
│   └── ui/                      Painel Streamlit
│
├── data/
│   ├── raw/                     PubMedQA como veio do Hugging Face
│   ├── processed/               Datasets anonimizados e curados
│   ├── sintetico/               Protocolos, FAQ, laudos e prontuários gerados
│   └── indices/                 Índices FAISS persistidos
│
├── models/
│   ├── adapters/                Adapter LoRA (versionado — ~50 MB)
│   └── gguf/                    Modelo quantizado (fora do Git — baixado do HF Hub)
│
├── notebooks/colab/             Notebooks de fine-tuning e exportação
├── scripts/                     Pontos de entrada de linha de comando
├── tests/                       Testes automatizados
├── docs/                        Relatório técnico, diagramas, gráficos, rastreabilidade
└── logs/                        Trilha de auditoria e traces (fora do Git)
```

---

## 7. Logging e auditoria

O requisito 3 do enunciado pede *"logging detalhado para rastreamento e auditoria"*. O
projeto grava em **três destinos simultâneos**, cada um com um público diferente.

### 7.1 Console — para quem está assistindo

Saída colorida, em tempo real, com um ícone por tipo de evento:

```
[14:16:20] INFO  ▶ Consulta recebida
           INFO  → guardrail_entrada · Iniciando
           INFO  🔒 guardrail_entrada · 1 identificador removido da pergunta
           INFO  🛡 guardrail_entrada · Concluido (41 ms)
           INFO  🗄 consultar_prontuario · Concluido (70 ms)
           INFO  📚 recuperar_evidencia · Concluido (92 ms)
           INFO  🧠 raciocinio_clinico · Concluido (125 ms)
           INFO  ⚕ guardrail_saida · Alergia a penicilina conflita com a classe sugerida
           INFO  🚨 guardrail_saida · Alerta emitido para a equipe medica
           INFO  👤 Risco acima do limiar - resposta retida para validacao medica
           INFO  ■ Consulta finalizada (aguardando_validacao) (374 ms)
```

### 7.2 `logs/app.log` — histórico legível

Texto corrido, com rotação em cinco arquivos de 5 MB. Inclui módulo, função e nível.

### 7.3 `logs/auditoria/auditoria-AAAA-MM-DD.jsonl` — a trilha formal

Um evento por linha, em JSON, particionado por dia. É o registro consultável por máquina:

```json
{
  "ts": "2026-08-23T17:16:21.235272+00:00",
  "nivel": "INFO",
  "mensagem": "🧠 raciocinio_clinico · Concluido (125 ms)",
  "trace_id": "6a8a087077284a30",
  "sequencia": 11,
  "tipo": "llm",
  "etapa": "raciocinio_clinico",
  "duracao_ms": 124.59,
  "conclusao": true,
  "dados": { "delta": { "resposta": "... conforme o protocolo interno [P2] ..." } }
}
```

Consultas diretas no terminal:

```bash
# Quantas vezes o guardrail de saída reprovou hoje?
jq 'select(.tipo=="guardrail")' logs/auditoria/auditoria-*.jsonl | jq -s length

# Todas as consultas que exigiram validação humana
jq -r 'select(.tipo=="validacao_humana") | .trace_id' logs/auditoria/*.jsonl
```

### 7.4 `logs/traces/<trace_id>.json` — o dossiê de uma consulta

O registro completo de **uma** consulta: metadados, configuração vigente (com segredos
mascarados), todos os eventos em ordem, latência por etapa e o desfecho. É o arquivo que o
painel Streamlit lê para desenhar a aba *Trilha do grafo*.

### 7.5 Como a instrumentação é garantida

Todo nó do grafo é decorado com `@instrumentar`, que registra automaticamente início, fim,
duração, chaves do estado recebido e delta produzido:

```python
@instrumentar("recuperar_evidencia", tipo=TipoEvento.RECUPERACAO)
def recuperar_evidencia(estado: EstadoClinico) -> EstadoClinico:
    ...
```

A alternativa — repetir blocos de log em cada nó — falharia no primeiro esquecimento, e um
nó sem rastro invalidaria a garantia de auditabilidade. Com o decorador, **auditar é o
padrão, não a exceção**.

### 7.6 Proteção de dados na trilha

Nenhum texto chega ao disco sem passar pelo redator de PII (`definir_redator`). Textos
longos são truncados com marcação explícita de quantos caracteres foram omitidos — a trilha
nunca mente por omissão. Segredos (`OPENAI_API_KEY`, `HF_TOKEN`) são mascarados antes de
serem gravados.

---

## 8. Controle de custo

Cada chamada a um modelo é contabilizada em tokens e em dólares, e o consumo entra na
trilha de auditoria. Um teto por sessão (`MAX_CUSTO_USD_SESSAO`) **bloqueia** chamadas
pagas ao ser atingido:

```
Modelo                        Chamadas  Tok.entrada  Tok.saida        US$
-------------------------------------------------------------------------
medgraph                             1         1420        310   0.000000
gpt-4o-mini                          3         4200        900   0.001170
-------------------------------------------------------------------------
TOTAL                                4         5620       1210   0.001170
Teto da sessao: US$ 1.00  |  Saldo: US$ 0.9988
```

O modelo local aparece com custo zero **de propósito**: o volume processado continua sendo
registrado, o que permite comparar, no relatório técnico, o custo por consulta entre o
modelo ajustado local e a API paga.

---

## 9. Dados utilizados

| Fonte | Natureza | Uso |
| --- | --- | --- |
| [PubMedQA](https://pubmedqa.github.io/) | Real, público | Fine-tuning e base de evidência científica |
| Protocolos internos | **Sintético** | RAG e fine-tuning |
| FAQ do corpo médico | **Sintético** | Fine-tuning |
| Modelos de laudo, receita e parecer | **Sintético** | Fine-tuning |
| Prontuários de pacientes | **Sintético** | Consulta estruturada |

### Nota de transparência

**Nenhum dado real de paciente é utilizado neste projeto.** Bases hospitalares reais não
são publicamente distribuíveis, e o próprio enunciado aceita *"dataset anonimizado ou
exemplo de dados sintéticos"*. Os prontuários, protocolos e documentos do Hospital Vida
Plena são gerados por script, com nomes fictícios.

Ainda assim, o pipeline de anonimização é aplicado sobre eles — demonstrando a técnica
exigida no item 1 do enunciado e garantindo que o mesmo código funcionaria sobre dados
reais, se houvesse.

---

## 10. Rastreabilidade dos requisitos

Cada exigência do enunciado recebeu um código (`REQ-1`, `REQ-3b`, `REQ-E1`…), catalogado em
[`src/medgraph/requisitos.py`](src/medgraph/requisitos.py) e citado nas docstrings do
código que a atende.

A matriz **requisito → arquivo → linha** é gerada automaticamente:

```bash
make rastreabilidade
```

Resultado em [`docs/rastreabilidade.md`](docs/rastreabilidade.md). O gerador também acusa
requisitos ainda sem cobertura e tags digitadas incorretamente.

---

## 11. Testes

```bash
make testes
```

A suíte cobre a validação da configuração, a integridade do catálogo de requisitos, os três
destinos de logging, o ciclo de vida da trilha de auditoria, o cálculo de custo, a trava de
orçamento e a consistência do arquivo de políticas — incluindo uma verificação de que
`conduta_terapeutica` **sempre** exige validação humana.

---

## 11.1 Documentação do projeto

| Documento | Conteúdo |
| --- | --- |
| [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) | Relatório técnico completo — arquitetura, fine-tuning, avaliação, defeitos encontrados e limitações declaradas |
| [`docs/rastreabilidade.md`](docs/rastreabilidade.md) | Matriz requisito → arquivo → linha, gerada do código |
| [`docs/guia_colab.md`](docs/guia_colab.md) | Passo a passo para executar o fine-tuning no Google Colab |
| [`docs/roteiro_video.md`](docs/roteiro_video.md) | Roteiro cronometrado do vídeo de entrega, com divisão entre os integrantes |
| [`docs/diagramas/`](docs/diagramas/) | Diagrama do grafo em PNG, Mermaid e ASCII |
| [`docs/graficos/`](docs/graficos/) | Gráficos da avaliação |

O relatório técnico é **gerado**: a narrativa está em `docs/relatorio_base.md` e os
números são lidos dos artefatos que o pipeline produziu. Um relatório com números
digitados à mão começa correto e envelhece errado — basta reexecutar a avaliação
para que a tabela deixe de corresponder aos arquivos, sem que ninguém perceba.

---

## 12. Licença e créditos

Projeto acadêmico desenvolvido para o Tech Challenge da Fase 3 do curso de pós-graduação
**8IADT — FIAP/Alura**.

- **PubMedQA** — Jin et al., 2019. [pubmedqa.github.io](https://pubmedqa.github.io/)
- **Llama 3.2** — Meta AI, sob a Llama 3.2 Community License
- **LangChain / LangGraph** — LangChain Inc., licença MIT
