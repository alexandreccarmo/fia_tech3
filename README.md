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

**`HuggingFaceEmbeddings`** (de `langchain-huggingface`) para transformar texto em vetor. Usamos o
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

O projeto tem treze dependências. Todas são instaladas pela célula 2 do
notebook; quem for rodar os testes na própria máquina usa o `requirements.txt`,
que traz a mesma lista.

Quatro delas são as duas seções acima, e ficam aqui pelo nome do pacote para que
a lista fique completa:

| Pacote | O que trouxemos dele |
| --- | --- |
| `langchain` | `ChatPromptTemplate` e `StrOutputParser` — o prompt como objeto e a saída como texto puro |
| `langchain-community` | `FAISS` como vector store |
| `langchain-huggingface` | `HuggingFacePipeline`, que transforma o modelo que treinamos num componente da cadeia |
| `langgraph` | `StateGraph`, `add_conditional_edges` e o desenho do diagrama |

As demais nove estão descritas abaixo, agrupadas por finalidade.

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
guardrail_entrada ──(recusado)──────────────────────────────────────┐
        │                                                            │
consultar_prontuario                                                 │
        │                                                            │
verificar_exames                                                     │
        │                                                            │
recuperar_evidencia                                                  │
        │                                                            │
responder (LLM)                                                      │
        │                                                            │
verificar_resposta ──(crítico)── emitir_alerta ── validacao_humana ──┤
        │                                                            │
        └────────────────────────────────────────────────────────────┴──> montar_resposta
```

O enunciado descreve o fluxo automatizado como um sistema que, "ao receber
informações sobre um paciente", pode "verificar exames pendentes, sugerir
tratamentos e emitir alertas para a equipe médica". As três etapas têm um nó
cada: `verificar_exames`, `responder` e `emitir_alerta`.

| Nó | O que faz |
| --- | --- |
| `guardrail_entrada` | Recusa pedidos fora do escopo |
| `consultar_prontuario` | Busca alergias, medicações, exames e comorbidades no SQLite |
| `verificar_exames` | Lê exames críticos e pendências antes de qualquer geração |
| `recuperar_evidencia` | Busca semântica nos protocolos e modelos de documento |
| `responder` | Chama a LLM ajustada, pela cadeia LangChain |
| `verificar_resposta` | Regras clínicas e procedência da citação |
| `emitir_alerta` | Endereça cada achado crítico à equipe do setor do paciente |
| `validacao_humana` | Retém a resposta — a execução termina aqui |
| `montar_resposta` | Monta o texto final com fontes, pendências, alertas e aviso |

---

## Como executar

### No Colab — o projeto completo, ~25 minutos

**[▶ Abrir o notebook no Google Colab](https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb)**

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb
```

#### Como esse link é formado

Não há integração configurada entre o repositório e o Colab, nem token, nem
aplicativo instalado. O link é apenas uma convenção de endereço: o Colab tem um
carregador de GitHub que responde a URLs no formato

```
https://colab.research.google.com/github/USUARIO/REPOSITORIO/blob/BRANCH/CAMINHO.ipynb
```

Compare com o endereço do mesmo arquivo no GitHub:

```
https://github.com/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/medgraph_lite.ipynb
```

É a mesma URL, trocando `github.com/` por `colab.research.google.com/github/`. O
Colab baixa o arquivo `.ipynb` — que é JSON versionado como qualquer outro — e o
abre no editor. Funciona sem login porque o repositório é público; se fosse
privado, o Colab pediria autorização de acesso ao GitHub.

**Alternativa sem montar a URL:** dentro do Colab, *Arquivo → Abrir notebook →
aba GitHub*, digite `alexandreccarmo/fia_tech3` e escolha o notebook na lista.

#### Como o código chega até lá

O notebook sozinho não traz o projeto. Quem traz é a célula da seção 3:

```python
if not os.path.isdir("/content/fia_tech3"):
    !git clone --depth 1 https://github.com/alexandreccarmo/fia_tech3.git /content/fia_tech3

sys.path.insert(0, "/content/fia_tech3")
```

Ela clona o repositório dentro da máquina virtual do Google e acrescenta o
diretório ao caminho de importação do Python. É por isso que a linha seguinte
consegue fazer `from medgraph_lite import ...` — o pacote veio no clone.

O caminho é de mão única: o Colab lê do GitHub e nunca escreve de volta sozinho.

#### Para trabalhar sobre uma cópia própria

Quem quiser modificar o projeto precisa que a célula 3 clone **o repositório
dele**, e não este — senão estaria executando o código de outra pessoa.

1. Publique a cópia no GitHub, como repositório **público**
2. Na célula da seção 3, aponte `REPO` para ela:
   ```python
   REPO = "https://github.com/SEU_USUARIO/SEU_REPO.git"
   ```
3. Monte o link trocando as duas partes do endereço:
   ```
   https://colab.research.google.com/github/SEU_USUARIO/SEU_REPO/blob/main/notebooks/medgraph_lite.ipynb
   ```

#### Executando

1. **Ambiente de execução → Alterar o tipo → T4 GPU**
2. Rode a seção 1 (verifica a GPU) e a 2 (instala as bibliotecas, ~3 min)
3. **Reinicie a sessão** e continue da seção 3
4. Na **seção 3.2**, escolha quanto quer treinar (veja abaixo)
5. Rode até o fim

#### Escolhendo o tamanho do treino

A seção 3.2 do notebook tem uma linha que decide quanto tempo tudo vai levar:

```python
PERFIL = "rapido"
```

O padrão é `"rapido"` — é o que permite percorrer o notebook inteiro numa aula.
Para trocar, mude a palavra e **reexecute a partir da seção 3.2**; as seções 1 e
2 não precisam ser repetidas enquanto a sessão estiver viva.

| `PERFIL` | Exemplos | Épocas | Passos | Treino | **Notebook inteiro** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `"rapido"` *(padrão)* | 398 | 1 | 24 | 1,8 min | **~12 min** |
| `"completo"` | 848 | 2 | 106 | ~7,8 min | **~18 min** |
| `"intensivo"` | 1.028 | 2 | 128 | ~9,4 min | **~20 min** |

O `"intensivo"` usa todo o `pqa_labeled` do PubMedQA, que tem 1.000 exemplos
anotados por especialistas — pedir mais do que isso não aumenta o conjunto.

Os tempos usam o ritmo que medimos numa T4: **4,4 segundos por passo**. Só o
treino muda de duração — as demais seções somam cerca de 10 minutos em qualquer
perfil.

**Qual escolher.** O `"rapido"` demonstra a técnica inteira e é suficiente para
apresentar o projeto. Ele treina de verdade, mas com 24 passos o modelo aprende a
forma da resposta sem consolidar a decisão — o que se vê nos [resultados](#resultados).

Use `"completo"` se quiser números melhores na comparação antes/depois. São seis
minutos a mais no total, e é o perfil que dá chance de o colapso de classe se
desfazer.

O `"intensivo"` só compensa se essa comparação for o foco da apresentação; para um
modelo de 0,5 bilhão, o ganho sobre o `"completo"` é pequeno.

Os avisos em vermelho do `pip`, na seção 2, são esperados: ele reclama de
pacotes que o Colab traz de fábrica e que não usamos.

| Seção | O que acontece | Tempo |
| --- | --- | ---: |
| 1–2 | GPU e dependências | 3 min |
| 3 | PubMedQA, protocolos, documentos internos, anonimização | 1 min |
| 4 | Base de prontuários | instantâneo |
| 5 | Avaliação do modelo original e **fine-tuning** | 4 min (perfil `rapido`) |
| 6 | Avaliação do modelo ajustado, comparação | 2 min |
| 7 | Índice de evidência | 1 min |
| 8 | Assistente com a cadeia LangChain | 1 min |
| 9 | Fluxo LangGraph: quatro consultas, quatro caminhos | 2 min |
| 10 | Conclusão | — |

### Na sua máquina — sem GPU

O treino precisa de GPU, mas todo o resto do projeto roda na sua máquina em
segundos: as regras clínicas, a anonimização, a consulta ao prontuário, o
roteamento do grafo e a trilha de auditoria. Usamos isso durante o
desenvolvimento inteiro — corrigir uma regra de segurança não deveria custar uma
sessão de Colab.

#### Pré-requisitos

| Item | Versão | Como conferir |
| --- | --- | --- |
| Python | 3.10 ou superior | `python3 --version` |
| git | qualquer | `git --version` |

Não é necessário GPU, nem conta em serviço nenhum. A instalação ocupa cerca de
1,5 GB, quase tudo do `torch`, que vem como dependência do `transformers`.

#### Passo 1 — clonar o repositório

```bash
git clone https://github.com/alexandreccarmo/fia_tech3.git
cd fia_tech3
```

#### Passo 2 — criar o ambiente

```bash
make setup
```

O alvo cria um ambiente virtual em `.venv/`, atualiza o `pip` e instala as
dependências do `requirements.txt` mais `pytest` e `ruff`. Leva de 2 a 5 minutos,
conforme a conexão — a maior parte é o download do `torch`.

Ao terminar, imprime `Pronto. Rode: make testes`.

**Se o `python3` da sua máquina for anterior ao 3.10** — o macOS ainda traz um
3.9 em `/usr/bin` — o alvo para antes de criar nada e diz qual versão encontrou.
Aponte para outro interpretador:

```bash
make setup PY_SETUP=/opt/homebrew/bin/python3
```

Sem essa checagem o `make setup` terminaria com sucesso e os testes falhariam
depois com `SyntaxError`, num arquivo que não tem nada de errado.

> **Por que um ambiente virtual.** As versões que o projeto usa não devem
> interferir no Python do sistema. Todos os alvos do `Makefile` chamam
> `.venv/bin/python` explicitamente, então não é preciso ativar nada.

#### Passo 3 — rodar os testes

```bash
make testes
```

São 63 testes, em menos de quatro segundos. A saída termina com:

```
63 passed, 1 warning in 1.10s
```

O aviso é uma biblioteca anunciando mudança futura numa API que não usamos
diretamente — o texto varia com a versão instalada. Pode ignorar.

Se algum teste falhar, o nome dele diz o que quebrou — cada teste cobre uma
afirmação específica, e a docstring explica por que aquilo importa.

#### Passo 4 — ver o fluxo funcionando

```bash
make demo
```

Este é o comando mais útil para entender o projeto sem abrir o Colab. Ele
executa as quatro consultas da apresentação e imprime, para cada uma:

- o **caminho percorrido no grafo**, nó a nó, com a latência de cada etapa;
- os **alertas levantados** pelos guardrails, com a severidade de cada um;
- a **resposta final**, com fontes e o aviso de que não substitui avaliação
  médica.

O que observar:

| Caso | O que deve acontecer |
| --- | --- |
| Conflito de alergia | Passa por `validacao_humana` e a resposta sai marcada como **`[RETIDA]`** |
| Interação de fármacos | Mesmo desfecho, por outro motivo — varfarina com amiodarona |
| Consulta simples | Percorre o caminho completo sem alertas |
| Pedido fora do escopo | Só dois nós: `guardrail_entrada` e `montar_resposta`. **Não chega à LLM** |

No fim, o comando informa quantos eventos a trilha de auditoria registrou.

> **O que é simulado aqui.** Apenas duas coisas: a geração de texto (as respostas
> são fixas) e a busca de evidência, que usa sobreposição de palavras em vez de
> embeddings — o demo roda sem GPU e sem baixar modelo. Guardrails, prontuário,
> regras clínicas, roteamento e auditoria executam exatamente como no Colab.

#### Comandos disponíveis

```bash
make ajuda
```

| Comando | O que faz |
| --- | --- |
| `make setup` | Cria o ambiente e instala as dependências |
| `make testes` | Roda a suíte |
| `make demo` | Executa o fluxo no terminal |
| `make lint` | Verifica estilo com o `ruff` |
| `make formatar` | Corrige estilo e ordena imports |
| `make limpar` | Remove caches e arquivos gerados |

#### Se algo falhar

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| `make: command not found` | `make` não instalado | No macOS, `xcode-select --install` |
| Erro na instalação do `torch` | Python 3.13 ou superior | O stack de ML ainda não tem *wheels*; use 3.10 a 3.12 |
| `ModuleNotFoundError: medgraph_lite` | Rodou o `pytest` direto, fora do `make` | Use `make testes` — ele ajusta o `PYTHONPATH` |
| Testes falham após alterar código | Comportamento mudou | O nome do teste indica o quê; a docstring explica por que aquilo é verificado |

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
├── data/               dataset sintético em JSONL (`make dataset`)
├── notebooks/
│   └── medgraph_lite.ipynb
├── docs/
│   └── relatorio_tecnico.md
├── tests/              63 testes, rodam sem GPU
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
de treino, acima da cota diária do Colab.

Com 0,5 bilhão, medimos **4,4 segundos por passo**: o treino padrão leva **1,8
minuto** e demonstra exatamente a mesma técnica. O QLoRA treina **8.798.208
parâmetros de 323.917.696 — 2,72% do modelo**.

O notebook oferece três perfis de treino, de 24 a 128 passos, para quem quiser
resultados mais representativos ao custo de alguns minutos a mais.

Ao final do treino o notebook **salva o adapter** em `/content/adapter`. São só
as matrizes A e B do LoRA — alguns megabytes, sem o modelo base — e é com esse
arquivo que o ajuste pode ser reaproveitado depois:
`PeftModel.from_pretrained(modelo_base, "/content/adapter")`.

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

## Resultados

Execução medida numa T4 do Colab, com o perfil padrão — 24 passos, 1,8 minuto de
treino:

| Sistema | Adesão ao formato | Acurácia |
| --- | ---: | ---: |
| Modelo base | 0% | 55% |
| Modelo ajustado | **90%** | **10%** |

> **Estes números são de uma execução anterior a duas correções na avaliação:**
> o conjunto de teste passou a ser separado antes da repetição do material do
> hospital (três dos vinte casos tinham cópias no treino), e a verificação de
> formato passou a aceitar as três famílias de marcador. As duas mudanças
> afetam as colunas acima. Reexecute o notebook e reporte **os números da sua
> execução** — a análise abaixo continua valendo, porque descreve o mecanismo,
> mas os valores exatos precisam vir da medição nova.

A leitura direta dessa tabela seria "o ajuste ensinou o formato e destruiu a
acurácia". Está parcialmente errada, e vale explicar as duas partes.

### A acurácia do modelo base estava inflada

A decisão era extraída procurando `yes`, `no` ou `maybe` em qualquer ponto da
resposta. O modelo base não segue formato nenhum — a adesão foi 0% — mas escreve
parágrafos em inglês, e um parágrafo em inglês contém a palavra "no" com
facilidade. Quando o rótulo esperado era `no`, isso contava como acerto.

Os 55% também se aproximam da frequência da classe `yes` no PubMedQA, que é a
marca de quem responde sempre a classe majoritária. As duas colunas mediam coisas
diferentes: uma, a decisão que o modelo declarou; a outra, a chance de uma palavra
aparecer num texto livre.

Corrigimos a medição: a extração passou a registrar se a decisão foi **declarada**
(o modelo escreveu `Decisao: X` na primeira linha) ou apenas **inferida** do
texto.

### O colapso de classe é real

Com 24 passos, o modelo aprendeu a estrutura e passou a responder sempre a mesma
decisão. Os 10% são próximos da frequência dessa classe no conjunto de teste, o
que confirma o diagnóstico.

É o comportamento esperado para um treino tão curto: exemplos suficientes para
memorizar três linhas, insuficientes para associar o conteúdo do contexto à
decisão certa. A avaliação passou a reportar a distribuição das respostas ao lado
da esperada, e o notebook avisa sozinho quando uma decisão responde por 80% ou
mais dos casos — sem esse diagnóstico, um modelo que não decide nada aparenta ter
desempenho.

### O que concluímos

**O fine-tuning cumpriu o que se propunha.** O objetivo era ensinar o formato, e a
adesão foi de 0% para 90%. Sem isso, os guardrails descritos abaixo não teriam
onde se apoiar — eles precisam localizar a decisão e a citação para verificá-las.

**E o treino curto não ensina a decisão.** Reportar apenas a melhora do formato
seria omitir metade do resultado. Melhorar a decisão é questão de mais dados e
mais passos, e não de mudança de método: trocar `PERFIL = "rapido"` por
`"completo"` na seção 3.2 do notebook leva o treino de 24 para 103 passos, ao
custo de seis minutos a mais. Veja
[Escolhendo o tamanho do treino](#escolhendo-o-tamanho-do-treino).

A análise completa está em [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md),
seção 7.

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

### Citação obrigatória, e conferida contra o que foi recuperado

Resposta sem fonte não passa. Mas exigir apenas que **haja** um marcador deixa
passar a falha mais provável de um modelo treinado a citar: inventar a citação.
O treino ensina o formato `... [E1]` com os exemplos do PubMedQA, e nada impede
o modelo de escrever `[E1]` respondendo sobre um protocolo do hospital — o
formato fica perfeito e a fonte não existe.

Por isso o guardrail compara o que foi citado com o que o RAG de fato entregou
naquela consulta. Citação a fonte não recuperada é achado crítico; citação
correta com um marcador extra ao lado é apenas atenção, porque a afirmação tem
procedência e o resto é ruído.

É o que torna a explicabilidade verificável: a fonte não é uma promessa do
modelo, é um campo que o sistema confere.

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

O nó `emitir_alerta` grava na mesma trilha o alerta **endereçado**: leva o setor
onde o paciente está internado, para que a auditoria consiga responder não só
"o que foi alertado" como "a quem". Alerta sem destinatário é linha de log.

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

63 testes, rodando em menos de três segundos, sem GPU.

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

## O vídeo

O entregável pede uma demonstração de **até 15 minutos** cobrindo quatro pontos:
o treinamento e o funcionamento da LLM personalizada, a execução de um fluxo
automatizado, respostas a perguntas clínicas contextualizadas, e os logs e a
validação das respostas.

Um roteiro que cabe no tempo, gravando a tela com o notebook já executado (não
espere o treino ao vivo — deixe a saída pronta e percorra as células):

| Tempo | Seção | O que mostrar |
| ---: | --- | --- |
| 0:00–1:30 | Abertura | O problema, o princípio ("o assistente nunca prescreve") e o mapa do notebook |
| 1:30–3:30 | §3 | Os três materiais do enunciado, a anonimização nos dois sentidos e a separação treino/teste |
| 3:30–6:30 | §5 | QLoRA: o que está congelado, o que treina, a contagem de parâmetros, a curva de perda e o adapter salvo |
| 6:30–8:00 | §6 | Antes × depois, e o diagnóstico de colapso de classe — inclusive o que **não** melhorou |
| 8:00–9:30 | §7–8 | RAG com marcador de fonte e a cadeia LangChain montada com `\|` |
| 9:30–13:00 | §9 | **O centro da demonstração**: as quatro consultas, a trilha colorida ao vivo e a figura do caminho percorrido |
| 13:00–14:30 | §9.3 | A trilha em JSONL, as consultas retidas e o alerta endereçado à equipe |
| 14:30–15:00 | Fecho | Limitações declaradas e o aviso de que é trabalho acadêmico |

Os dois momentos que mais rendem: a consulta recusada **saltando** do guardrail
direto para a resposta final sem tocar na LLM, e o conflito de alergia sendo
**retido** com o alerta emitido. Os dois se veem na figura do caminho percorrido,
sem precisar de explicação.

Para ensaiar sem gastar cota de GPU: `make demo` roda o fluxo inteiro no
terminal, com o modelo simulado.

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
| Dataset sintético e anonimizado | `dados.py`, exportado em `data/*.jsonl` |
| Relatório técnico | `docs/relatorio_tecnico.md` |
| Vídeo de até 15 minutos | roteiro em [O vídeo](#o-vídeo) — a gravar |

---

## Sobre os dados

Nenhum dado real de paciente é usado. Prontuários, protocolos, documentos e FAQ
do Hospital Vida Plena foram escritos por nós. Ainda assim aplicamos o pipeline
de anonimização sobre eles, para demonstrar a técnica e garantir que o mesmo
código funcionaria com dados reais.

O material está versionado em `data/`, um registro por linha em JSONL — quem for
conferir o trabalho abre o arquivo, em vez de ler `dados.py`. Para reexportar
depois de mexer nos dados:

```bash
make dataset
```

O PubMedQA não é redistribuído aqui: é público, tem licença própria e o notebook
o baixa pelo `datasets`.

Este é um trabalho acadêmico. Não passou por comitê de ética nem por validação
clínica, e não deve ser usado em assistência a pacientes.
