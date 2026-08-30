# MedGraph Lite

**Tech Challenge — Fase 3 · Pós-Tech 8IADT**
Hospital Vida Plena (cenário fictício)

| Integrante | RM |
| --- | --- |
| Alexandre Carneiro do Carmo | 370980 |
| Brunno Costa Castigrini | 371429 |
| Pedro Henrique Azevedo Aragão | 373481 |
| Valter Willian de Oliveira Filho | 370979 |

---

## O que construímos

Um assistente que responde dúvidas clínicas usando uma LLM que nós mesmos
ajustamos, consultando o prontuário do paciente e os protocolos do hospital, e
que **para de funcionar de propósito** quando a resposta esbarra num risco.

O projeto inteiro roda em um notebook do Google Colab, em cerca de 25 minutos,
incluindo o fine-tuning. Não é preciso criar conta em serviço nenhum.

### A regra que orientou tudo

O assistente não prescreve. Ele mostra a evidência, diz de onde ela veio, e a
decisão continua sendo do médico.

Essa frase parece protocolar, mas foi ela que definiu a arquitetura. Se o
sistema pode recusar-se a responder, alguém precisa decidir quando — e essa
decisão não pode estar no prompt, porque prompt se contorna. Ela está na
topologia do grafo: existe um nó do qual a execução não sai sem intervenção
humana.

---

## Por que LangChain

O LangChain resolve dois problemas que apareceriam de qualquer jeito, e nós
usamos ele exatamente nesses dois pontos.

### O que usamos dele

**`FAISS` como vector store** (`langchain_community.vectorstores`). Os
protocolos do hospital precisam ser encontrados por significado, não por palavra
exata: quem pergunta "que antibiótico usar em quem tem alergia a penicilina"
precisa chegar ao protocolo de betalactâmicos, que não contém nenhuma dessas
palavras na pergunta. O FAISS roda em memória, é salvo em disco e não exige
serviço externo — foi o mesmo que vimos nas aulas.

**`HuggingFaceEmbeddings`** para transformar texto em vetor. Usamos o
`all-MiniLM-L6-v2`, que é pequeno e roda rápido.

**`ChatPromptTemplate`** para o prompt do assistente. Poderíamos ter montado a
mensagem com f-string, e no começo foi assim. Trocamos porque o prompt é a peça
que mais muda durante o desenvolvimento, e quando ele está espalhado em
concatenações pelo código ninguém consegue revisar o que o modelo está de fato
recebendo. Como template, ele fica num lugar só, com as variáveis explícitas.

**`HuggingFacePipeline`** para embrulhar o modelo que treinamos. É o que
transforma um modelo carregado em memória num componente que o LangChain
entende.

**Composição com `|`** — o operador que encadeia as peças:

```python
cadeia = ChatPromptTemplate | HuggingFacePipeline | StrOutputParser
```

### Por que isso importa na prática

O ganho concreto apareceu na hora de avaliar. Precisávamos rodar o **mesmo**
fluxo duas vezes: uma com o modelo original e outra com o modelo ajustado, para
comparar. Como a LLM é só mais um elo da cadeia, trocá-la não exigiu mexer em
mais nada — nem no prompt, nem na recuperação, nem no grafo.

Se a chamada ao modelo estivesse escrita à mão no meio do fluxo, essa
comparação teria custado uma refatoração.

---

## Por que LangGraph

O LangChain encadeia passos que acontecem sempre na mesma ordem. Nosso problema
não é esse.

Uma pergunta fora do escopo não deve chegar ao modelo. Uma resposta que conflita
com alergia registrada não deve ser entregue. Uma consulta sem paciente
vinculado pula a etapa de prontuário. São **caminhos diferentes**, decididos
durante a execução, e é isso que o LangGraph resolve.

### O que usamos dele

**`StateGraph`** com um estado tipado (`TypedDict`) que atravessa todos os nós.
Cada nó recebe o estado, devolve o estado modificado, e o que um nó produz fica
visível para os seguintes. Isso é o que permite, por exemplo, que a verificação
de alergia enxergue tanto o prontuário (lido no nó 2) quanto a resposta gerada
(produzida no nó 4).

**`add_conditional_edges`** nos dois pontos onde o fluxo se ramifica:

- depois do guardrail de entrada — pedido recusado vai direto para a resposta,
  sem passar pelo modelo;
- depois da verificação — conflito crítico desvia para validação humana.

**`add_node` / `add_edge` / `set_entry_point`** para montar o resto, e `compile()`
para gerar o executável.

**`get_graph().draw_mermaid_png()`** para desenhar o diagrama. Não é acessório:
uma das exigências do trabalho é apresentar o fluxo, e ter o desenho gerado a
partir do grafo real evita que a documentação descreva um fluxo que o código não
segue mais.

### O que ganhamos com isso

O ponto que nos convenceu foi a validação humana. Com o LangGraph, "a execução
para e espera um médico" deixou de ser uma frase no relatório e virou um nó com
uma aresta condicional chegando nele. Dá para apontar no diagrama.

E há um efeito colateral que só percebemos depois: como o caminho é dado, ele
pode ser registrado. A trilha de auditoria de cada consulta é, literalmente, a
lista de nós por onde ela passou.

---

## Outras dependências usadas

Além do LangChain e do LangGraph, descritos acima, o projeto usa as bibliotecas
abaixo. Todas são instaladas pela célula 2 do notebook; quem for rodar os testes
na própria máquina usa o `requirements.txt`, que traz a mesma lista.

### Fine-tuning

| Biblioteca | Para que serve aqui |
| --- | --- |
| `transformers` | Carrega o modelo base e o tokenizador, e faz a geração de texto. É a biblioteca sobre a qual todo o resto do treino se apoia |
| `peft` | Implementa o LoRA — cria as matrizes treináveis ao lado das camadas congeladas e cuida de aplicá-las durante o treino |
| `bitsandbytes` | Faz a quantização em 4 bits (a parte "Q" do QLoRA), incluindo o formato NF4 e a dupla quantização |
| `trl` | Fornece o `SFTTrainer`, o laço de treino supervisionado. Poupa escrever à mão o preparo dos lotes, o cálculo da perda e o acúmulo de gradiente |
| `accelerate` | Distribui o modelo entre GPU e CPU conforme a memória disponível. É o que faz `device_map="auto"` funcionar |
| `datasets` | Baixa o PubMedQA do Hugging Face e organiza os exemplos no formato que o `SFTTrainer` espera |

### Recuperação de evidência

| Biblioteca | Para que serve aqui |
| --- | --- |
| `sentence-transformers` | Fornece o modelo `all-MiniLM-L6-v2`, que converte cada protocolo e cada pergunta em vetor. É o que permite buscar por significado |
| `faiss-cpu` | Guarda esses vetores e responde "quais textos mais se parecem com este". Roda em memória, sem servidor |

### Apresentação

| Biblioteca | Para que serve aqui |
| --- | --- |
| `matplotlib` | Desenha as cinco figuras: curva de perda, comparação antes e depois, caminho percorrido no grafo, linha do tempo e alertas por severidade |

### Da biblioteca padrão

Estas não aparecem no `requirements.txt` porque já vêm com o Python, mas vale
registrar o papel de cada uma:

| Módulo | Para que serve aqui |
| --- | --- |
| `sqlite3` | Guarda a base de prontuários. Escolhido por ser embutido, sem servidor a subir |
| `contextvars` | Leva a trilha de auditoria até os nós do grafo sem passá-la pelo estado — o `TypedDict` do LangGraph descarta chaves não declaradas |
| `json` | Escreve e lê a trilha em JSONL |
| `logging` | Log de sistema, separado da trilha das consultas |
| `unicodedata` | Normaliza acentuação antes de comparar texto nos guardrails |
| `re` | Padrões de anonimização e verificação de formato da resposta |

---

## O fluxo

```
guardrail_entrada ──(recusado)────────────────────┐
        │                                          │
consultar_prontuario                               │
        │                                          │
recuperar_evidencia                                │
        │                                          │
responder (LLM)                                    │
        │                                          │
verificar_resposta ──(crítico)── validacao_humana ─┤
        │                                          │
        └──────────────────────────────────────────┴──> montar_resposta
```

| Nó | O que faz |
| --- | --- |
| `guardrail_entrada` | Recusa pedidos fora do escopo |
| `consultar_prontuario` | Busca alergias, medicações, exames e comorbidades no SQLite |
| `recuperar_evidencia` | Busca semântica nos protocolos e modelos de documento |
| `responder` | Chama a LLM ajustada, pela cadeia LangChain |
| `verificar_resposta` | Regras clínicas e exigência de citação |
| `validacao_humana` | Retém a resposta — a execução termina aqui |
| `montar_resposta` | Monta o texto final com fontes e aviso |

---

## Como executar

### No Colab — o projeto completo, ~25 minutos

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb
```

1. **Ambiente de execução → Alterar o tipo → T4 GPU**
2. Rode a seção 1 (verifica a GPU) e a 2 (instala as bibliotecas, ~3 min)
3. **Reinicie a sessão** e continue da seção 3
4. Rode até o fim

Os avisos em vermelho do `pip`, na seção 2, são esperados: ele reclama de
pacotes que o Colab traz de fábrica e que não usamos.

| Seção | O que acontece | Tempo |
| --- | --- | ---: |
| 1–2 | GPU e dependências | 3 min |
| 3 | PubMedQA, protocolos, documentos internos, anonimização | 1 min |
| 4 | Base de prontuários | instantâneo |
| 5 | Avaliação do modelo original e **fine-tuning** | 10 min |
| 6 | Avaliação do modelo ajustado, comparação | 2 min |
| 7 | Índice de evidência | 1 min |
| 8 | Assistente com a cadeia LangChain | 1 min |
| 9 | Fluxo LangGraph: quatro consultas, quatro caminhos | 2 min |
| 10 | Conclusão | — |

### Na sua máquina — sem GPU, em segundos

Serve para conferir a lógica de segurança, o prontuário e o roteamento sem
depender do Colab.

```bash
git clone https://github.com/alexandreccarmo/fia_tech3.git
cd fia_tech3
make setup
```

```bash
make testes
```

```bash
make demo
```

O `make demo` percorre os quatro casos da apresentação e imprime, para cada um,
o caminho no grafo, os alertas levantados e a resposta final. Só a geração de
texto é simulada — guardrails, prontuário, recuperação e roteamento executam de
verdade. Usamos isso para ensaiar sem gastar a cota de GPU do Colab.

---

## Estrutura

```
fia_tech3/
├── medgraph_lite/
│   ├── dados.py        PubMedQA, protocolos, modelos de documento, anonimização
│   ├── prontuario.py   base SQLite e a entidade Paciente
│   ├── treino.py       configuração do QLoRA e formato do prompt de treino
│   ├── rag.py          índice FAISS e recuperação com marcador de fonte
│   ├── chain.py        pipeline LangChain: prompt → LLM → parser
│   ├── guardrails.py   limites de atuação e regras clínicas
│   ├── grafo.py        fluxo LangGraph
│   ├── auditoria.py    trilha por consulta e logger de sistema
│   └── graficos.py     as cinco figuras
├── notebooks/
│   └── medgraph_lite.ipynb
├── docs/
│   └── relatorio_tecnico.md
├── tests/              47 testes, rodam sem GPU
├── demo.py             o fluxo no terminal
├── Makefile
└── requirements.txt
```

---

## O fine-tuning

Treinamos o **Qwen2.5-0.5B-Instruct** com QLoRA sobre PubMedQA, os protocolos do
hospital, o FAQ do corpo médico e os modelos de laudo, receita e procedimento.

O enunciado deixa a escolha do modelo livre. Escolhemos um modelo pequeno por
uma razão prática: numa versão anterior deste projeto medimos 42 segundos por
passo com um modelo de 3 bilhões de parâmetros na T4 gratuita — quase seis horas
de treino, acima da cota diária do Colab. Com 0,5 bilhão, o treino leva cerca de
oito minutos e demonstra a mesma técnica.

**O que o ajuste ensina não é medicina.** É o formato da resposta:

```
Decisao: yes|no|maybe
<justificativa em até 3 frases, apoiada apenas no contexto>
[P1]
```

Isso não é preciosismo. Os guardrails precisam localizar a decisão e a citação
para conseguir verificá-las. Um modelo que responde bem, mas cada vez de um
jeito, inviabiliza toda a checagem que vem depois.

Por isso avaliamos duas métricas separadas: **adesão ao formato** e **acurácia**.
"Errou a resposta" e "não seguiu o formato" são problemas diferentes, com
correções diferentes — uma métrica única esconderia isso.

---

## Segurança

### Alergia por classe, não por nome

Um paciente alérgico a penicilina e uma resposta sugerindo ceftriaxona não têm
nenhuma semelhança de texto. Comparar strings não acusa nada. As duas são
betalactâmicas, e reatividade cruzada é conhecimento farmacológico — por isso
mantemos uma tabela de classes.

### Evitação medida por frase

Quando o assistente escreve "evitar penicilina devido à alergia registrada", ele
está acertando. Tratar isso como se fosse prescrição gera alarme falso — e um
médico que vê alerta crítico toda vez que o sistema acerta passa a ignorar
alertas críticos.

Detectamos o contexto da menção, mas a janela é a **frase**, não uma quantidade
de caracteres ao redor. Em "Evitar penicilina. Iniciar ceftriaxona.", uma janela
por proximidade alcançaria o "evitar" da frase anterior e liberaria uma sugestão
real de fármaco contraindicado — errando na direção que uma regra de segurança
não pode errar.

### Citação obrigatória

Resposta sem fonte não passa. É o que torna a explicabilidade verificável: a
fonte não é uma promessa do modelo, é um campo que o sistema confere.

---

## Logs e auditoria

Três registros, com públicos diferentes:

| Destino | Conteúdo |
| --- | --- |
| Console | Uma linha por etapa, com ícone e latência — para quem acompanha a execução |
| `auditoria.jsonl` | Um evento por linha, em JSON — para auditar depois |
| `medgraph.log` | Eventos de sistema: carga de modelo, construção do índice |

Cada consulta recebe um `trace_id`. Sem ele, os eventos de consultas diferentes
se misturam no arquivo e deixa de ser possível reconstruir o que aconteceu em
cada uma.

```json
{
  "ts": "2026-08-30T20:14:07.512830+00:00",
  "trace_id": "a3f9c1e08b42",
  "sequencia": 5,
  "nivel": "CRITICO",
  "etapa": "verificar_resposta",
  "detalhe": "2 achado(s), 1 critico(s)",
  "ms": 0.82
}
```

Escolhemos JSONL porque o arquivo pode ser consultado sem escrever nenhum
parser:

```bash
jq -r 'select(.etapa=="validacao_humana") | .trace_id' auditoria.jsonl | sort -u | wc -l
```

```bash
jq -s 'group_by(.etapa) | map({etapa: .[0].etapa, ms: (map(.ms) | add)})' auditoria.jsonl
```

A seção 9.3 do notebook lê essa trilha e mostra quais consultas foram retidas e
por quê.

---

## Os gráficos

Todos em matplotlib, dentro do notebook:

1. **Curva de perda** — o treino convergiu
2. **Antes e depois** — o que o ajuste mudou, em adesão ao formato e acurácia
3. **Caminho percorrido no grafo** — o fluxo desenhado uma vez por consulta, com
   os nós visitados em destaque e os demais apagados
4. **Linha do tempo** — as mesmas trilhas com a latência de cada nó
5. **Alertas por severidade** — quantos achados cada nível produziu

A terceira é a que mais ajuda numa apresentação: ver o pedido recusado saltar do
guardrail direto para a resposta final, sem tocar na LLM, dispensa explicar o que
é roteamento condicional.

---

## Testes

47 testes, rodando em menos de três segundos, sem GPU.

| Área | O que verificamos |
| --- | --- |
| Anonimização | Remove os identificadores **e preserva** valores de exame |
| Guardrail de entrada | Recusa pedidos impróprios, inclusive com acentuação |
| Regras clínicas | Alergia por classe, evitação por frase, interação, citação |
| Prontuário | Consulta, paciente inexistente, exame crítico |
| Grafo | Os quatro caminhos, a parada para validação, a trilha |
| Auditoria | Gravação em disco, separação por consulta, arquivo truncado |
| LangChain | Composição da cadeia e preenchimento do prompt |
| Notebook | JSON válido e células com sintaxe correta |

O que não é coberto automaticamente é o treino e a qualidade das respostas —
ambos exigem GPU. Isso é verificado no próprio notebook, comparando o modelo
antes e depois do ajuste.

---

## Documentação

| Documento | Conteúdo |
| --- | --- |
| Este README | Execução, arquitetura e as decisões que tomamos |
| [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) | Relatório técnico: fine-tuning, assistente, avaliação, limitações |
| Notebook | O projeto executando, seção a seção |

---

## Requisitos do enunciado

| Requisito | Onde |
| --- | --- |
| Fine-tuning com protocolos do hospital | `dados.PROTOCOLOS`, notebook §5 |
| Perguntas frequentes de médicos | `dados.FAQ` |
| Modelos de laudos, receitas e procedimentos | `dados.DOCUMENTOS` |
| Preprocessing, anonimização e curadoria | `dados.anonimizar`, notebook §3.1 |
| LangChain integrando a LLM customizada | `chain.py`, notebook §8 |
| Consultas a base estruturada | `prontuario.py`, notebook §4 |
| Contextualização com dados do paciente | `grafo.no_responder` |
| Limites de atuação | `guardrails.py` |
| Logging para rastreamento e auditoria | `auditoria.py`, notebook §9.3 |
| Explicabilidade por fonte | `rag.py` + `guardrails.verificar_resposta` |
| Projeto modularizado | pacote `medgraph_lite/` |
| Fluxos do LangGraph | `grafo.py` |
| Dataset sintético e anonimizado | `dados.py` |
| Relatório técnico | `docs/relatorio_tecnico.md` |

---

## Sobre os dados

Nenhum dado real de paciente é usado. Prontuários, protocolos, documentos e FAQ
do Hospital Vida Plena foram escritos por nós. Ainda assim aplicamos o pipeline
de anonimização sobre eles, para demonstrar a técnica e garantir que o mesmo
código funcionaria com dados reais.

Este é um trabalho acadêmico. Não passou por comitê de ética nem por validação
clínica, e não deve ser usado em assistência a pacientes.
