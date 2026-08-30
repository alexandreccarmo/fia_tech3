# Relatório Técnico — MedGraph Lite

**Tech Challenge · Fase 3 · Pós-Tech 8IADT**
Hospital Vida Plena (cenário fictício)

| Integrante | RM |
| --- | --- |
| Alexandre Carneiro do Carmo | 370980 |
| Brunno Costa Castigrini | 371429 |
| Pedro Henrique Azevedo Aragão | 373481 |
| Valter Willian de Oliveira Filho | 370979 |

---

## Sumário

1. O problema e a decisão central
2. Arquitetura
3. Dados e preparo
4. Fine-tuning
5. O assistente com LangChain
6. Segurança e validação
7. Avaliação e análise dos resultados
8. Limitações declaradas
9. Rastreabilidade dos requisitos

---

## 1. O problema e a decisão central

O enunciado pede um assistente virtual médico treinado com dados próprios do
hospital, capaz de auxiliar em condutas clínicas, responder dúvidas do corpo
médico e sugerir procedimentos com base em protocolos internos — coordenando
fluxos de decisão automatizados e seguros.

Há uma tensão embutida nesse pedido. Um sistema que "sugere condutas" e um
sistema que "nunca prescreve sem validação humana" parecem incompatíveis, e a
forma como essa tensão é resolvida define o projeto inteiro.

### A decisão

> O assistente **nunca prescreve**. Ele apresenta evidência, aponta a fonte de
> cada afirmação e devolve a decisão ao médico responsável.

Isso não é uma instrução no prompt — é comportamento verificado em código. Um
prompt pode ser contornado; um nó de grafo que interrompe a execução, não.

Quando a resposta conflita com uma alergia registrada, quando há interação
medicamentosa relevante, ou quando falta citação de fonte, a execução **para** e
aguarda um médico. A seção 6 detalha como.

### Por que uma versão enxuta

Este projeto substituiu uma versão anterior, mais extensa, que levava quase três
horas para executar e dependia de conversão GGUF, `llama.cpp`, Ollama e Hugging
Face Hub. Numa única tarde de testes, essa cadeia quebrou em quatro pontos
distintos — nenhum deles relacionado ao trabalho em si: verificação de GPU em
runtime sem GPU, conflito de versão de `torchao`, memória esgotada durante a
fusão, e o `llama.cpp` recusando o tokenizer de uma versão nova do
`transformers`.

Para um trabalho que precisa ser **apresentado**, cada dependência externa é um
risco sem contrapartida. A versão atual roda inteira em um notebook, em cerca de
25 minutos, sem conta em serviço nenhum. A versão anterior está preservada em
`modelo_cancelado/`.

---

## 2. Arquitetura

```
                     Médico (corpo clínico)
                              |
                              v
        +------------------------------------------+
        |     LangGraph — fluxo de decisão         |
        |     7 nós, 2 pontos de desvio            |
        +------------------------------------------+
           |          |            |            |
           v          v            v            v
        SQLite     FAISS         LLM        auditoria
      prontuário  protocolos   ajustada      JSONL
       exames     documentos  (LangChain)
```

Cada dependência abaixo do grafo é substituível sem tocar no fluxo. O grafo
recebe a função de resposta por parâmetro — é o que permite rodar o **mesmo**
fluxo com o modelo base e com o ajustado, que é exatamente o que a avaliação
comparativa precisa.

### Módulos

| Módulo | Responsabilidade |
| --- | --- |
| `dados.py` | PubMedQA, protocolos, modelos de documento, anonimização |
| `prontuario.py` | Base SQLite e a entidade `Paciente` |
| `treino.py` | Configuração do QLoRA e formato do prompt de treino |
| `rag.py` | Índice FAISS e recuperação com marcador de fonte |
| `chain.py` | Pipeline LangChain: prompt → LLM → parser |
| `guardrails.py` | Limites de atuação e regras clínicas |
| `grafo.py` | Fluxo LangGraph |
| `auditoria.py` | Trilha por consulta e logger de sistema |
| `graficos.py` | As cinco figuras da apresentação |

Cerca de 1.200 linhas de Python.

---

## 3. Dados e preparo

### 3.1 Fontes

| Fonte | Natureza | Uso |
| --- | --- | --- |
| PubMedQA | Real, público | Fine-tuning e evidência científica |
| 5 protocolos internos | **Sintético** | Fine-tuning e RAG |
| 3 modelos de documento (laudo, receita, procedimento) | **Sintético** | Fine-tuning e RAG |
| 8 perguntas frequentes | **Sintético** | Fine-tuning |
| 3 prontuários | **Sintético** | Consulta estruturada |

**Nenhum dado real de paciente é utilizado.** Bases hospitalares reais não são
publicamente distribuíveis, e o próprio enunciado aceita "dataset anonimizado ou
exemplo de dados sintéticos".

### 3.2 Anonimização

O pipeline é aplicado mesmo sobre os dados sintéticos, demonstrando a técnica e
garantindo que o mesmo código funcionaria sobre dados reais.

Remove cinco classes de identificador: nome próprio precedido de marcador
(`paciente`, `Sr.`, `Dra.`), CPF, telefone e número de prontuário.

**O cuidado central é o inverso do óbvio.** Testar se o anonimizador remove o
identificador cobre metade do problema. A outra metade — e a mais perigosa — é
ele apagar o que não devia: um anonimizador que remove `Lactato 4.5 mmol/L`
entrega texto limpo e clinicamente inútil, e essa falha passa despercebida
porque o texto continua parecendo correto.

Por isso a suíte testa nos dois sentidos: 7 padrões que devem sair, 6 exemplos
de dado clínico que devem permanecer intactos.

**Uma armadilha registrada.** O padrão de nome próprio distingue
`paciente Joao Silva` de `paciente deve incluir` pelas classes de caixa
(`[A-Z]`, `[a-zà-ú]`). Usar `re.IGNORECASE` para aceitar `Dra.` além de `dra.`
anularia essas classes — a flag vale para a expressão inteira — e o padrão
passaria a casar frases comuns, mutilando o texto. As variantes maiúsculas estão
escritas explicitamente no padrão, e há um teste de regressão para isso.

### 3.3 Curadoria

O conjunto de treino combina ~350 exemplos do PubMedQA com o material do
hospital repetido seis vezes. A repetição é deliberada: são 8 pares de pergunta
e resposta contra centenas de exemplos científicos, e sem peso eles não
influenciariam o formato aprendido.

---

## 4. Fine-tuning

### 4.1 O que está sendo ensinado

**Não é medicina.** O modelo base já viu literatura biomédica no pré-treino. O
que o ajuste ensina é o **formato**:

```
Decisao: yes|no|maybe
<justificativa em até 3 frases, apoiada apenas no contexto>
[P1]
```

Esse formato não é estético. Os guardrails precisam **encontrar** a decisão e a
citação para poder verificá-las — um modelo que responde bem mas em formato
livre quebra tudo o que vem depois.

### 4.2 Escolha do modelo base

**Qwen2.5-0.5B-Instruct.** O enunciado deixa a escolha livre ("como LLaMA,
Falcon ou um outro").

| Critério | Razão |
| --- | --- |
| Aberto | Sem licença *gated* para esperar aprovação |
| 0,5 bilhão | Treina em ~8 min numa T4; um modelo de 3B levaria ~6 h |
| Suporte a português | Adequado aos protocolos internos |

A versão anterior deste projeto usou Llama-3.2-3B e mediu **42 segundos por
passo** numa T4 gratuita — 484 passos, quase 6 horas, acima da cota diária do
Colab. Foi essa medição que motivou a troca.

### 4.3 QLoRA

Treinar um modelo de bilhões de parâmetros exige guardar, além dos pesos, os
gradientes e os estados do otimizador. O QLoRA resolve isso em dois movimentos:

**LoRA** congela o modelo e adiciona, ao lado de certas camadas, duas matrizes
finas cujo produto representa o ajuste. Treinam-se apenas elas — cerca de 1% dos
parâmetros.

**Quantização** carrega o modelo base em 4 bits, no formato **NF4**
(*NormalFloat4*), otimizado para pesos com distribuição normal. Com dupla
quantização, comprimem-se também as constantes de quantização.

Os adaptadores são aplicados às **sete projeções**: atenção (`q`, `k`, `v`, `o`)
e MLP (`gate`, `up`, `down`). Aplicar apenas em `q` e `v`, como é comum em
tutoriais, rende menos quando a tarefa muda o **estilo** da resposta — que é
exatamente o caso aqui.

### 4.4 Configuração

| Parâmetro | Valor |
| --- | --- |
| `r` / `alpha` / `dropout` | 16 / 32 / 0,05 |
| `max_seq_length` | 512 |
| Lote efetivo | 16 (4 × 4 de acúmulo) |
| Épocas | 1 |
| Taxa de aprendizado | 2e-4, com escalonamento cosseno |

### 4.5 Compatibilidade de biblioteca

A API do `trl` renomeou argumentos várias vezes. A configuração é montada
**tentando construir e removendo apenas o que a biblioteca recusar**, em vez de
descobrir por introspecção quais nomes ela aceita.

A diferença importa: a introspecção pode errar para menos, e errar para menos
aqui significa treinar com uma configuração diferente da pedida, em silêncio.
Deixar a própria biblioteca decidir elimina essa classe de erro.

---

## 5. O assistente com LangChain

### 5.1 O pipeline

A LLM ajustada é embrulhada em `HuggingFacePipeline` e composta com o operador
`|`, como nos exemplos das aulas:

```python
ChatPromptTemplate  |  HuggingFacePipeline  |  StrOutputParser
```

Chamar `modelo.generate()` diretamente produziria a mesma resposta. A diferença
não é burocrática: o prompt vira objeto versionável em vez de f-string
espalhada, e trocar a LLM — ajustada, base, ou uma API — não mexe no resto do
fluxo.

### 5.2 Consulta à base estruturada

O nó `consultar_prontuario` busca no SQLite e devolve um objeto `Paciente`, cujo
`resumo()` entra no prompt antes do contexto recuperado. É assim que a resposta
fica contextualizada com dados atualizados: alergias, medicações em uso, exames
críticos e comorbidades.

### 5.3 A convenção de citação

Cada trecho chega ao prompt **já etiquetado** com seu marcador — `[P1]`, `[D2]`.
O modelo cita o marcador que recebeu, e o guardrail de saída confere se a
resposta traz alguma citação.

Isso é o que torna a explainability verificável em vez de declarada: a fonte não
é uma promessa do modelo, é um campo que o sistema checa.

---

## 6. Segurança e validação

### 6.1 Guardrail de entrada

Recusa pedidos fora do escopo — "pule a validação humana", "prescreva direto",
"assine o atestado". A comparação é feita sobre texto **normalizado**, sem
acento nem caixa: os padrões são escritos em ASCII e ninguém digita assim.
Escrever cada padrão nas duas grafias funcionaria e teria de ser lembrado em
toda adição futura — uma proteção que depende de memória já está quebrada.

### 6.2 Regras clínicas

**Alergia por classe, não por nome.** Um paciente alérgico a penicilina e uma
resposta sugerindo ceftriaxona não produzem semelhança textual alguma. A tabela
de classes farmacológicas resolve: as duas são betalactâmicas, e a reatividade
cruzada é conhecimento farmacológico, não similaridade de string.

**Evitação medida por frase, não por proximidade.** Quando o assistente escreve
"evitar penicilina devido à alergia registrada", ele está acertando — alertar
como se estivesse prescrevendo produz fadiga de alarme, e um médico que vê
alerta crítico toda vez que o sistema acerta aprende a ignorar alertas críticos.

Mas a janela de detecção precisa ser a **oração**, não um número de caracteres:
em "Evitar penicilina. Iniciar ceftriaxona.", uma janela por proximidade
alcançaria o "evitar" da frase anterior e rebaixaria uma sugestão real de
fármaco contraindicado. O erro apontaria na direção inaceitável para uma regra
de segurança.

A menção em contexto de evitação é **rebaixada** para severidade informativa, e
não descartada: se a heurística errar, o pior que acontece é um alerta discreto
onde deveria haver um grave — e ele continua visível.

**Interações e valores críticos.** Pares de relevância reconhecida (varfarina
com amiodarona, sulfametoxazol ou fluconazol) e exames marcados como críticos no
prontuário. Um teste garante que toda interação cadastrada usa fármacos que o
detector reconhece — sem isso, a regra existiria no código e seria inerte.

### 6.3 Guardrail de saída

Exige citação de fonte. Sem ela, a resposta não é verificável e não passa.

### 6.4 Validação humana

Havendo qualquer achado crítico, o fluxo desvia para `validacao_humana`, a
resposta é marcada como **retida** e a execução termina sem liberá-la. É o item
do enunciado que diz "nunca prescrever diretamente, sem validação humana",
implementado como topologia de grafo.

### 6.5 Logging e auditoria

Dois registros, com públicos diferentes:

| Destino | Conteúdo | Para quem |
| --- | --- | --- |
| Console | Uma linha colorida por etapa, com ícone e latência | Quem assiste à execução |
| `auditoria.jsonl` | Um evento por linha, em JSON | Quem audita depois |
| `medgraph.log` | Eventos de sistema — carga de modelo, índice | Diagnóstico |

Cada consulta recebe um `trace_id`. Sem ele, os eventos de consultas diferentes
se misturariam no arquivo e a trilha deixaria de reconstruir o que aconteceu em
cada uma — que é o propósito dela.

O evento traz carimbo de tempo, sequência, etapa, nível (`INFO`, `ALERTA`,
`CRITICO`), duração e detalhe. O formato JSONL é consultável por máquina sem
parser próprio.

**Uma decisão de implementação que vale registrar.** A trilha é passada aos nós
por `contextvars`, e não pelo estado do grafo. O estado do LangGraph é um
`TypedDict`, e o framework **descarta em silêncio** qualquer chave não
declarada: o objeto simplesmente não chegava aos nós, a consulta rodava inteira
e a trilha saía vazia, sem nenhum aviso. Declarar o objeto no estado resolveria
o descarte e criaria outro problema — o estado atravessa serialização, e uma
trilha com arquivo aberto não é serializável.

---

## 7. Avaliação e análise dos resultados

### 7.1 Metodologia

Vinte casos separados do conjunto de treino, **nunca vistos pelo modelo**. O
mesmo conjunto é avaliado duas vezes: com o modelo base e com o ajustado.

Duas métricas, medidas **separadamente**:

| Métrica | O que mede |
| --- | --- |
| **Adesão ao formato** | A resposta começa com `Decisao:` e cita uma fonte? |
| **Acurácia** | A decisão (`yes`/`no`/`maybe`) está correta? |

Separá-las importa porque as causas e as correções são diferentes. "Errou a
resposta" pede mais dados ou mais épocas; "não seguiu o formato" pede ajuste do
prompt de treino. Uma métrica única confundiria as duas.

### 7.2 Resultados

_Os números desta seção são produzidos pelo notebook, seções 5.1 e 6, e devem
ser transcritos aqui após a execução. A tabela abaixo mostra o formato._

| Sistema | Adesão ao formato | Acurácia |
| --- | ---: | ---: |
| Modelo base | _a preencher_ | _a preencher_ |
| Modelo ajustado | _a preencher_ | _a preencher_ |

**O que esperar.** A adesão ao formato deve subir de forma marcante — é o que o
ajuste ensina. A acurácia deve subir pouco ou ficar estável: um modelo de 0,5
bilhão treinado por 8 minutos não aprende medicina, e não é isso que se pretende
aqui. Anunciar ganho de acurácia como resultado principal seria vender o que o
método não entrega.

### 7.3 Figuras

| Figura | Responde a |
| --- | --- |
| Curva de perda | O treino convergiu? |
| Antes × depois | O que o ajuste melhorou? |
| **Caminho percorrido no grafo** | O fluxo decide alguma coisa? |
| Linha do tempo | Onde está o custo de latência? |
| Achados por severidade | Os limites funcionam? |

A terceira é a mais direta: desenha o grafo uma vez por consulta, com os nós
visitados em destaque. O pedido recusado saltando do guardrail direto para a
resposta final, sem tocar na LLM, dispensa qualquer explicação sobre roteamento
condicional.

---

## 8. Limitações declaradas

Um sistema clínico que não declara o alcance da sua verificação induz uma
confiança que não merece.

| Limitação | Consequência | O que seria necessário |
| --- | --- | --- |
| Modelo de 0,5 B, treino de ~8 min | Capacidade de raciocínio limitada | Modelo maior e treino longo, fora do escopo de uma demonstração |
| Tabela de fármacos restrita aos protocolos incluídos | Fármaco ausente **não gera alerta** | Base farmacológica licenciada |
| Interações limitadas a pares reconhecidos | Interações não catalogadas passam | Base de interações licenciada |
| Detecção de nome próprio é heurística, não NER | Nome incomum pode escapar | Modelo de NER em português clínico |
| Prontuários sintéticos | Desempenho em dado real é desconhecido | Validação em ambiente controlado |
| Avaliação em 20 casos | Intervalo de confiança largo | Conjunto de teste maior |
| Sem validação clínica | Não apto a uso assistencial | Estudo prospectivo com supervisão médica |

**Este é um projeto acadêmico.** Não foi submetido a comitê de ética, não passou
por validação clínica e não deve ser utilizado em assistência a pacientes.

---

## 9. Rastreabilidade dos requisitos

| Requisito do enunciado | Onde é atendido |
| --- | --- |
| Fine-tuning de LLM com protocolos do hospital | `dados.PROTOCOLOS`, `treino.py`, notebook §5 |
| ...com perguntas frequentes de médicos | `dados.FAQ` |
| ...com modelos de laudos, receitas e procedimentos | `dados.DOCUMENTOS` |
| Preprocessing, anonimização e curadoria | `dados.anonimizar`, notebook §3.1 |
| LangChain integrando a LLM customizada | `chain.py`, notebook §8 |
| Consultas a base estruturada | `prontuario.py`, notebook §4 |
| Contextualização com dados do paciente | `grafo.no_responder` |
| Limites de atuação | `guardrails.py`, notebook §9 |
| Logging para rastreamento e auditoria | `auditoria.py`, notebook §9.3 |
| Explainability por fonte | `rag.py` + `guardrails.verificar_resposta` |
| Projeto modularizado em Python | pacote `medgraph_lite/` |
| Instruções completas no README | `README.md` |
| Fluxos do LangGraph | `grafo.py` |
| Dataset anonimizado ou sintético | `dados.py` |
| Relatório técnico | este documento |
| Vídeo de até 15 minutos | a gravar |

### Suíte de testes

47 testes, executando em menos de 3 segundos, sem GPU. Cobrem anonimização nos
dois sentidos, limites de atuação, regras clínicas, prontuário, os quatro
caminhos do grafo, a trilha de auditoria, o pipeline LangChain e a integridade
do notebook.

O que **não** é coberto automaticamente: o treino e a qualidade das respostas.
Ambos exigem GPU, e são verificados no próprio notebook, comparando o modelo
antes e depois do ajuste.

---

## Referências

- **PubMedQA** — Jin, Q. et al. *PubMedQA: A Dataset for Biomedical Research
  Question Answering.* EMNLP 2019.
- **QLoRA** — Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized
  LLMs.* NeurIPS 2023.
- **LoRA** — Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.*
  ICLR 2022.
- **Qwen2.5** — Qwen Team, Alibaba Cloud, 2024.
- **LangChain / LangGraph** — LangChain Inc., licença MIT.
