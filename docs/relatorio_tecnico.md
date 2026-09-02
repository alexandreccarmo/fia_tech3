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
25 minutos, sem conta em serviço nenhum.

---

## 2. Arquitetura

```
                     Médico (corpo clínico)
                              |
                              v
        +------------------------------------------+
        |     LangGraph — fluxo de decisão         |
        |     9 nós, 2 pontos de desvio            |
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
| `dados.py` | PubMedQA, protocolos, modelos de documento, anonimização, separação treino/teste |
| `prontuario.py` | Base SQLite e a entidade `Paciente` |
| `treino.py` | Configuração do QLoRA e formato do prompt de treino |
| `rag.py` | Índice FAISS e recuperação com marcador de fonte |
| `chain.py` | Pipeline LangChain: prompt → LLM → parser |
| `guardrails.py` | Limites de atuação e regras clínicas |
| `grafo.py` | Fluxo LangGraph |
| `auditoria.py` | Trilha por consulta e logger de sistema |
| `graficos.py` | As cinco figuras da apresentação |

Cerca de 1.700 linhas de Python.

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

### 3.4 Separação entre treino e teste

Vinte exemplos ficam fora do treino e servem à avaliação da seção 7. A separação
acontece **antes** de o material do hospital ser repetido, e a ordem é o ponto
todo.

Na primeira versão o corte era feito depois: repetíamos o material seis vezes,
embaralhávamos tudo e ficávamos com as últimas vinte linhas. Como as cópias são
idênticas, **três dos vinte casos de teste tinham gêmeos no conjunto de treino**
— a avaliação media memorização e reportava generalização, e nada na saída
denunciava isso.

O conjunto de teste sai inteiro do PubMedQA, onde cada exemplo é único, e o
notebook confere a separação com um `assert` em vez de prometê-la num
comentário.

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

### 4.4 Configuração e resultado medido

| Parâmetro | Valor |
| --- | --- |
| `r` / `alpha` / `dropout` | 16 / 32 / 0,05 |
| `max_seq_length` | 512 |
| Lote efetivo | 16 (4 × 4 de acúmulo) |
| Épocas | 1 |
| Taxa de aprendizado | 2e-4, com escalonamento cosseno |

Execução medida numa T4 do Colab, com o perfil padrão:

| Medida | Valor |
| --- | ---: |
| Exemplos de treino | 378 |
| Passos | 24 |
| **Parâmetros treináveis** | **8.798.208** |
| Parâmetros congelados | 315.119.488 |
| **Proporção treinada** | **2,72%** |
| Ritmo | **4,4 s por passo** |
| Duração | **1,8 min** |
| Perda: primeiro → último passo | 2,96 → 1,60 |

A proporção de 2,72% fica abaixo do que se estimaria pela conta de uma projeção
quadrada isolada (3,6%). A razão é a **atenção com agrupamento de chaves** que o
Qwen2.5 emprega: `k_proj` e `v_proj` projetam para uma dimensão bem menor que
`q_proj`, porque várias cabeças de consulta compartilham o mesmo par de chave e
valor. O adaptador dessas projeções acompanha a dimensão menor e contribui com
menos parâmetros.

**Perfis disponíveis.** O notebook permite trocar o tamanho do treino. Os tempos
abaixo usam o ritmo medido de 4,4 s por passo:

| Perfil | Exemplos | Épocas | Passos | Treino |
| --- | ---: | ---: | ---: | ---: |
| `rapido` (padrão) | 398 | 1 | 24 | 1,8 min |
| `completo` | 848 | 2 | 106 | ~7,8 min |
| `intensivo` | 1.028 | 2 | 128 | ~9,4 min |

### 4.5 Compatibilidade de biblioteca

A API do `trl` renomeou argumentos várias vezes. A configuração é montada
**tentando construir e removendo apenas o que a biblioteca recusar**, em vez de
descobrir por introspecção quais nomes ela aceita.

A diferença importa: a introspecção pode errar para menos, e errar para menos
aqui significa treinar com uma configuração diferente da pedida, em silêncio.
Deixar a própria biblioteca decidir elimina essa classe de erro.
### 4.6 Um argumento recusado pela biblioteca

A camada de compatibilidade informou, na execução medida:

```
argumentos recusados por esta versao do trl: ['max_seq_length', 'warmup_ratio']
```

O primeiro é troca de nome, e está coberto: passamos o comprimento de sequência
pelos dois nomes possíveis, e a versão instalada aceitou `max_length`.

O segundo é uma perda efetiva — o treino roda sem aquecimento da taxa de
aprendizado. Com 24 passos e `warmup_ratio` de 0,03, o aquecimento duraria menos
de um passo, então não há consequência prática nesta configuração. Num treino
mais longo, convém verificar se a versão instalada aceita `warmup_steps`.

Registramos isso porque a alternativa seria não registrar: o argumento foi
descartado e o treino seguiu. É exatamente o tipo de divergência silenciosa entre
a configuração pedida e a aplicada que a camada de compatibilidade existe para
tornar visível.


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
O modelo cita o marcador que recebeu, e o guardrail de saída confere duas
coisas: que há citação, e que o marcador citado é um dos que foram de fato
recuperados naquela consulta (seção 6.3).

Isso é o que torna a explainability verificável em vez de declarada: a fonte não
é uma promessa do modelo, é um campo que o sistema checa.

### 5.4 O fluxo de decisão, nó a nó

O entregável pede o diagrama do fluxo. Ele é gerado pelo próprio grafo compilado
na seção 9 do notebook (`app.get_graph().draw_mermaid_png()`), o que impede que a
documentação descreva um fluxo que o código já não segue. Em texto:

```
guardrail_entrada ──(recusado)──────────────────────────────────────┐
        │                                                            │
consultar_prontuario                                                 │
        │                                                            │
verificar_exames                                                     │
        │                                                            │
recuperar_evidencia                                                  │
        │                                                            │
responder (LLM, pela cadeia LangChain)                               │
        │                                                            │
verificar_resposta ──(crítico)── emitir_alerta ── validacao_humana ──┤
        │                                                            │
        └────────────────────────────────────────────────────────────┴──> montar_resposta
```

| Nó | O que faz | Requisito |
| --- | --- | --- |
| `guardrail_entrada` | Recusa pedidos fora do escopo | 3 — limites de atuação |
| `consultar_prontuario` | Lê o SQLite do paciente | 2 — base estruturada |
| `verificar_exames` | Exames críticos e pendências, antes da LLM | E1 — "verificar exames pendentes" |
| `recuperar_evidencia` | Busca semântica com marcador de fonte | 3 — explainability |
| `responder` | Cadeia LangChain com a LLM ajustada | 2 — "sugerir tratamentos" |
| `verificar_resposta` | Regras clínicas e procedência da citação | 3 — limites e explainability |
| `emitir_alerta` | Endereça o achado crítico ao setor do paciente | E1 — "emitir alertas" |
| `validacao_humana` | Retém a resposta; a execução termina aqui | 3 — "nunca sem validação humana" |
| `montar_resposta` | Texto final com fontes, pendências, alertas e aviso | 3 — explainability |

**Por que um grafo, e não uma cadeia.** A cadeia da seção 5.1 executa sempre os
mesmos passos na mesma ordem. Uma pergunta fora do escopo não deve chegar ao
modelo; uma resposta em conflito com alergia registrada não deve ser entregue.
São caminhos decididos em execução, e é para isso que existem as duas arestas
condicionais.

**As três etapas que o enunciado nomeia têm um nó cada.** Ele descreve o fluxo
automatizado como um sistema que, ao receber informações sobre um paciente, pode
"verificar exames pendentes, sugerir tratamentos e emitir alertas para a equipe
médica": `verificar_exames`, `responder` e `emitir_alerta`.

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

**Interações.** Pares de relevância reconhecida — varfarina com amiodarona,
sulfametoxazol ou fluconazol. Um teste garante que toda interação cadastrada usa
fármacos que o detector reconhece: sem isso, a regra existiria no código e seria
inerte na prática.

**Exames críticos e pendentes.** Estes não olham o texto gerado, e sim o
paciente. Por isso saíram de `verificar_resposta` e passaram a viver no nó
`verificar_exames`, que roda **antes** da recuperação e antes da LLM. Pendência
de exame é informação que já está no prontuário desde antes da pergunta, e
fazê-la depender da resposta significava não verificá-la quando a resposta não
viesse.

### 6.3 Guardrail de saída: citação com procedência

Exige citação de fonte — e confere a **procedência** dela.

Verificar apenas que existe um marcador entre colchetes deixa passar a falha
mais provável de um modelo treinado a citar: inventar a citação. O treino ensina
o formato `... [E1]` com os exemplos do PubMedQA, e nada impede o modelo de
escrever `[E1]` respondendo sobre um protocolo do hospital. O formato fica
perfeito e a fonte não existe — que é exatamente o que a citação deveria
impedir.

O guardrail recebe do nó de recuperação os marcadores efetivamente entregues
naquela consulta e compara:

| Situação | Severidade |
| --- | --- |
| Nenhuma citação | Crítico |
| Só cita fonte que não foi recuperada | Crítico |
| Cita fonte recuperada, com marcador extra ao lado | Atenção |
| Cita apenas fontes recuperadas | Aprovado |

A terceira linha é uma escolha: a afirmação tem procedência e o resto é ruído.
Tratá-la como crítica transformaria excesso de citação em consulta retida.

**Um defeito que isso corrigiu.** A verificação anterior aceitava apenas as
famílias `[P..]` e `[E..]`. Os modelos de laudo, receita e procedimento entram
no índice com marcador `[D1]` a `[D3]`, e o próprio treino ensina o modelo a
citá-los: uma resposta correta, citando a fonte certa, era marcada como "sem
citação de fonte", virava achado crítico e era retida. O erro apontava para a
direção que produz fadiga de alarme.

### 6.4 Alerta endereçado e validação humana

Havendo qualquer achado crítico, o fluxo desvia para `emitir_alerta` e de lá
para `validacao_humana`. A resposta é marcada como **retida** e a execução
termina sem liberá-la. É o item do enunciado que diz "nunca prescrever
diretamente, sem validação humana", implementado como topologia de grafo.

O alerta leva destinatário: o setor onde o paciente está internado. Alerta sem
destinatário é linha de log — quem age sobre um conflito na UTI é a equipe da
UTI, e a auditoria precisa conseguir responder não só "o que foi alertado" como
"a quem".

### 6.5 Logging e auditoria

Três registros, com públicos diferentes:

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

Vinte casos separados do conjunto de treino, **nunca vistos pelo modelo** — a
separação descrita na seção 3.4, feita antes da repetição do material do
hospital e verificada por `assert` no notebook. O mesmo conjunto é avaliado duas
vezes: com o modelo base e com o ajustado.

Duas métricas, medidas **separadamente**:

| Métrica | O que mede |
| --- | --- |
| **Adesão ao formato** | A resposta começa com `Decisao:` e cita uma fonte? |
| **Acurácia** | A decisão (`yes`/`no`/`maybe`) está correta? |

Separá-las importa porque as causas e as correções são diferentes. "Errou a
resposta" pede mais dados ou mais épocas; "não seguiu o formato" pede ajuste do
prompt de treino. Uma métrica única confundiria as duas.

### 7.2 Resultados

Execução medida, perfil `rapido` (24 passos, 1,8 min):

| Sistema | Adesão ao formato | Acurácia |
| --- | ---: | ---: |
| Modelo base | 0% | 55% |
| Modelo ajustado | **90%** | **10%** |

> **Estes números precedem duas correções na avaliação**, descritas nas seções
> 3.4 e 7.4: a separação treino/teste passou a ser feita antes da repetição do
> material do hospital, e a verificação de formato passou a aceitar as três
> famílias de marcador. As duas afetam as colunas acima. O mecanismo analisado a
> seguir continua valendo — ele descreve *por que* os números se comportam assim
> —, mas os valores a reportar são os da execução nova.

A leitura direta dessa tabela — "o ajuste melhorou o formato e destruiu a
acurácia" — está parcialmente errada, e a parte errada é instrutiva.

### 7.3 Análise: por que a acurácia caiu

Há duas causas, e apenas uma delas é do modelo.

**Primeira: a acurácia do modelo base estava inflada pela medição.**

A decisão era extraída procurando `yes`, `no` ou `maybe` em qualquer ponto da
resposta. O modelo base não segue formato algum — a adesão foi 0% — mas produz
parágrafos em inglês, e um parágrafo em inglês contém a palavra "no" com
facilidade. Quando o rótulo esperado era `no`, isso contava como acerto.

Os 55% também são suspeitos por outro motivo: aproximam-se da frequência da
classe `yes` no PubMedQA. É a marca característica de quem responde sempre a
classe majoritária — o que produz acurácia sem produzir decisão.

Ou seja, as duas colunas não mediam a mesma coisa. Uma media a decisão que o
modelo declarou; a outra, a chance de uma palavra aparecer num texto livre.

**Segunda: o modelo ajustado colapsou em uma classe.**

Esta é real. Com 24 passos, o modelo aprendeu a estrutura da resposta e passou a
respondê-la sempre com a mesma decisão — `maybe`, no caso observado. A acurácia
de 10% é próxima da frequência de `maybe` no conjunto de teste, o que confirma o
diagnóstico.

O comportamento é esperado para um treino tão curto. O modelo teve exemplos
suficientes para memorizar a **forma** — três linhas, decisão, justificativa,
citação — e insuficientes para associar o conteúdo do contexto à decisão certa.

### 7.4 O que foi corrigido na avaliação

A extração da decisão passa a registrar sua **origem**:

| Origem | Significado |
| --- | --- |
| `declarada` | O modelo escreveu `Decisao: X` na primeira linha, como o formato exige |
| `inferida` | Não há formato; a palavra foi encontrada em algum ponto do texto |
| `ausente` | Nenhuma das três palavras apareceu |

E a avaliação passa a reportar a **distribuição das respostas**, ao lado da
distribuição esperada. É o que revela colapso de classe: um modelo que responde
sempre a mesma coisa tem acurácia igual à frequência daquela classe, sem ter
decidido nada — e sem a distribuição, esse número passa por desempenho.

O notebook emite um aviso automático quando uma única decisão responde por 80% ou
mais dos casos.

Duas outras correções entraram depois:

**A adesão ao formato só reconhecia dois tipos de marcador.** A verificação
exigia `[E1]` ou `[P..]`, e uma resposta citando corretamente um modelo de
documento (`[D2]`) contava como fora do formato. A métrica passou a aceitar
qualquer família de marcador — a mesma regra que o guardrail da seção 6.3 usa.

**O conjunto de teste tinha cópias no treino.** Três dos vinte casos, conforme a
seção 3.4. Enquanto isso valeu, parte da adesão medida era memorização.

### 7.5 Conclusão da avaliação

**O fine-tuning cumpriu o que se propôs.** O objetivo declarado era ensinar o
formato, e a adesão foi de 0% para 90% — o modelo passou a produzir respostas
que o sistema consegue verificar. Sem isso, os guardrails da seção 6 não teriam
onde se apoiar.

**O treino curto não ensina a decisão, e não deveria ser vendido como se
ensinasse.** Vinte e quatro passos sobre 378 exemplos são suficientes para a
forma e insuficientes para o conteúdo. Reportar a queda de acurácia como se fosse
apenas ruído de medição seria omitir metade do resultado.

Para melhorar a decisão, o caminho é mais dados e mais passos — o notebook
oferece perfis de 103 e 191 passos. É uma questão de minutos de GPU, não de
mudança de método.

### 7.6 Figuras

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
| Verificar exames pendentes | `grafo.no_verificar_exames` |
| Emitir alertas para a equipe médica | `grafo.no_emitir_alerta` |
| Limites de atuação | `guardrails.py`, notebook §9 |
| Logging para rastreamento e auditoria | `auditoria.py`, notebook §9.3 |
| Explainability por fonte | `rag.py` + `guardrails.verificar_resposta` |
| Projeto modularizado em Python | pacote `medgraph_lite/` |
| Instruções completas no README | `README.md` |
| Fluxos do LangGraph | `grafo.py` |
| Dataset anonimizado ou sintético | `dados.py`, exportado em `data/*.jsonl` |
| Diagrama do fluxo | seção 5.4, e gerado pelo grafo no notebook §9 |
| Relatório técnico | este documento |
| Vídeo de até 15 minutos | roteiro no README, seção "O vídeo" — a gravar |

### Suíte de testes

63 testes, executando em menos de 3 segundos, sem GPU. Cobrem anonimização nos
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
