# Relatório Técnico — MedGraph

**Assistente Clínico Auditável**
Tech Challenge — Fase 3 · Pós-Tech 8IADT

| Integrante | RM |
| --- | --- |
| Alexandre Carneiro do Carmo | 370980 |
| Brunno Costa Castigrini | 371429 |
| Pedro Henrique Azevedo Aragão | 373481 |
| Valter Willian de Oliveira Filho | 370979 |

> **Documento gerado automaticamente** a partir dos artefatos do repositório.
> A narrativa está em `docs/relatorio_base.md`; os números são lidos dos arquivos
> que o pipeline produziu. Para regenerar: `make relatorio`.
> Última geração: {{DATA}}

---

## Sumário

1. [O problema e a decisão central](#1-o-problema-e-a-decisão-central)
2. [Arquitetura](#2-arquitetura)
3. [Dados](#3-dados)
4. [Fine-tuning](#4-fine-tuning)
5. [O assistente](#5-o-assistente)
6. [Segurança e validação](#6-segurança-e-validação)
7. [Avaliação e análise dos resultados](#7-avaliação-e-análise-dos-resultados)
8. [Defeitos encontrados e o que aprendemos](#8-defeitos-encontrados-e-o-que-aprendemos)
9. [Limitações declaradas](#9-limitações-declaradas)
10. [Rastreabilidade dos requisitos](#10-rastreabilidade-dos-requisitos)

---

## 1. O problema e a decisão central

O enunciado pede um assistente virtual médico treinado com dados do hospital,
capaz de auxiliar em condutas clínicas e de organizar **fluxos de decisão
automatizados e seguros**.

A palavra que orientou o projeto inteiro foi **seguros**. Um assistente que
responde bem 95% das vezes e sugere um antibiótico ao qual o paciente tem
anafilaxia nos outros 5% não é 95% útil — é inutilizável. A partir dessa
constatação, adotamos um princípio que aparece em cada decisão de projeto:

> **O assistente nunca prescreve. Ele apresenta evidência, aponta a fonte de cada
> afirmação e devolve a decisão ao médico.**

A consequência técnica é que **as garantias de segurança não são pedidas ao
modelo, são impostas pelo sistema**. Um modelo de linguagem pode ser instruído a
citar fontes e, na maior parte das vezes, obedecer. "Na maior parte das vezes"
não é uma garantia. No MedGraph:

- a citação de fonte é **verificada** por expressão regular antes da entrega;
- o conflito com alergia é **detectado** por uma tabela farmacológica, não pela
  interpretação do modelo;
- a validação humana em casos de alto risco é uma **interrupção da execução**, não
  um aviso no texto.

O modelo redige. O sistema verifica. São papéis diferentes.

---

## 2. Arquitetura

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
   │        → emitir_alertas → validação_humana → montar_resposta        │
   └──────┬──────────────────┬───────────────────┬──────────────────┬────┘
          │                  │                   │                  │
          ▼                  ▼                   ▼                  ▼
   ┌────────────┐    ┌──────────────┐    ┌───────────────┐   ┌────────────┐
   │  SQLite    │    │ FAISS (RAG)  │    │ LLM ajustada  │   │  Auditoria │
   │ prontuários│    │ PubMedQA +   │    │ Llama-3.2-3B  │   │  JSONL +   │
   │  exames    │    │ protocolos   │    │ QLoRA/Ollama  │   │  traces    │
   └────────────┘    └──────────────┘    └───────────────┘   └────────────┘
```

O diagrama do fluxo, gerado pelo próprio LangGraph, está em
[`docs/diagramas/grafo.png`](diagramas/grafo.png), com as versões em
[ASCII](diagramas/grafo_ascii.txt) e [Mermaid](diagramas/grafo.mmd).

### Decisões técnicas

| Decisão | Alternativa descartada | Por quê |
| --- | --- | --- |
| Fine-tuning com **QLoRA no Colab** | Fine-tuning completo; treino local | 9 GB de VRAM contra 48; adapter de 50 MB contra 6 GB; a T4 gratuita treina em ~1 h o que o Apple Silicon levaria várias |
| Modelo servido pelo **Ollama** | `transformers` + MPS local | ~35 tok/s contra ~8; roda offline; é o mesmo padrão da Aula 05, então o código LangChain não muda |
| **Embeddings locais** (`multilingual-e5-small`) | `text-embedding-3-small` | Cobre inglês e português no mesmo espaço vetorial, custa zero e permite reconstruir o índice dezenas de vezes durante o desenvolvimento |
| **FAISS** | Chroma, Qdrant | Mesmo *vector store* das aulas, persistente, sem serviço externo |
| **Políticas em YAML** | Regras em código | Governança clínica não deveria exigir leitura de Python; mudança de política vira diff auditável |
| **Regras clínicas determinísticas** | Delegar ao LLM | Um modelo pode identificar que ceftriaxona é betalactâmico — e pode esquecer. Nenhuma dessas falhas seria detectável olhando a resposta |
| Auditoria por **`contextvars`** | Passar `trace_id` por parâmetro | Mantém os nós com assinatura limpa `(estado) -> estado` |
| **`interrupt_before`** na validação | Marcar "pendente" no estado | Marcar e seguir seria teatro: a resposta chegaria ao médico. Com interrupção, a execução para de verdade |

---

## 3. Dados

### 3.1 Fontes e o papel de cada uma

| Fonte | Natureza | Volume | Papel |
| --- | --- | --- | --- |
| PubMedQA `pqa_labeled` | Real, anotado por especialistas | 1.000 | Qualidade; única fonte da classe `maybe`; origem do conjunto de teste |
| PubMedQA `pqa_artificial` | Real, rótulos automáticos | 211.269 | Volume de treino; ensina o formato da tarefa |
| Protocolos internos | **Sintético** | 15 documentos | Base de RAG e domínio em português |
| FAQ do corpo médico | **Sintético** | 200 pares | Idioma e citação de protocolo interno |
| Modelos de documento | **Sintético** | 10 gabaritos | Formato institucional e limite da prescrição |
| Prontuários | **Sintético** | 40 pacientes | Consulta estruturada e regras de segurança |

> **Nota de transparência.** Nenhum dado real de paciente é utilizado. Bases
> hospitalares reais não são publicamente distribuíveis, e o enunciado aceita
> *"dataset anonimizado ou exemplo de dados sintéticos"*. Ainda assim, o pipeline
> de anonimização roda sobre todo o material — é ele que teria de funcionar se os
> dados fossem reais.

### 3.2 Anonimização

O módulo reconhece onze categorias de identificador: CPF, RG, Cartão Nacional de
Saúde, telefone, e-mail, CEP, data de nascimento, número de prontuário, CRM, nome
de pessoa e idade acima de 89 anos.

Duas políticas de substituição:

- **Mascarar** — `[CPF]`, `[NOME]`. Irreversível. Usada na trilha de auditoria,
  onde tokens estáveis permitiriam correlacionar consultas de um mesmo paciente —
  exatamente a reidentificação que se quer impedir.
- **Pseudonimizar** — `[NOME_7c1a]`, com sufixo derivado de HMAC. O mesmo valor
  sempre gera o mesmo token, o que preserva a **coerência referencial** do texto:
  "Maria Aparecida Souza", "Souza" e "Maria" recebem o mesmo pseudônimo, e o
  modelo continua aprendendo que se trata da mesma pessoa sem saber quem é.

**O que deliberadamente não é anonimizado:** valores laboratoriais, doses, sinais
vitais e datas de exame. São dados clínicos, não identificadores. Há teste
garantindo que `potássio 6.8 mEq/L` e `Ceftriaxona 2 g EV 1x/dia` sobrevivem
intactos — um anonimizador que apaga valor de exame entrega dado limpo e
clinicamente inútil, e a falha passa despercebida.

### 3.3 Curadoria

{{DADOS}}

### 3.4 Base de prontuários

{{PRONTUARIOS}}

### 3.5 Índice vetorial

{{INDICE}}

Os 500 exemplos de teste ficam **fora** do índice. Se estivessem lá, o assistente
poderia recuperar o próprio abstract da pergunta em avaliação, e a Seção 7 estaria
medindo recuperação em vez de raciocínio.

---

## 4. Fine-tuning

### 4.1 Por que QLoRA

| | Fine-tuning completo | QLoRA |
| --- | --- | --- |
| VRAM necessária (3B) | ~48 GB | **~9 GB** |
| Parâmetros treinados | 3,2 bilhões | **~24 milhões** (0,7%) |
| Artefato gerado | ~6 GB | **~50 MB** |
| Cabe na T4 gratuita | não | **sim** |

Os 50 MB do adapter cabem no Git — o resultado do treino fica versionado junto
com o código que o produziu.

### 4.2 Configuração

O modelo base é carregado em 4 bits com **NF4** e dupla quantização. NF4 é o tipo
de dado proposto no artigo do QLoRA: otimizado para pesos que seguem distribuição
normal, perde menos qualidade que o int4 comum. A dupla quantização comprime
também as constantes de quantização, economizando ~0,4 bit por parâmetro — margem
relevante quando se tem 16 GB.

Os adaptadores são aplicados às **sete projeções** — atenção (`q`, `k`, `v`, `o`)
e MLP (`gate`, `up`, `down`). Aplicar apenas em `q_proj`/`v_proj`, como é comum em
tutoriais, rende menos quando a tarefa muda o **estilo** da resposta — e é o nosso
caso: queremos que o modelo passe a responder num formato rígido, com decisão na
primeira linha e citação no fim.

### 4.3 Resultado do treino

{{TREINO}}

### 4.4 O dataset e a classe ausente

O subconjunto artificial do PubMedQA **não tem nenhum exemplo `maybe`** — os
rótulos automáticos só produzem `yes`/`no`. Todo o `maybe` do treino vem dos 449
exemplos de especialista, onde representa 11%. Num dataset final de milhares de
exemplos, `maybe` ficaria abaixo de 1%.

A consequência é mensurável: o modelo simplesmente nunca preveria `maybe`, e como
a avaliação usa macro-F1 — que dá o mesmo peso às três classes —, uma classe com
F1 zero limita a métrica a cerca de 0,67 **por construção**.

Corrigimos com repetição inversamente proporcional à frequência da classe
(`yes` ×2, `no` ×3, `maybe` ×8), elevando `maybe` a 9,6% do dataset — próximo dos
11% do conjunto de teste. Preferimos repetir a classe rara a descartar a comum
porque o conjunto de especialista já é pequeno e é o dado de maior qualidade que
temos. O risco assumido é memorização dos 49 exemplos repetidos oito vezes; ele é
monitorado pela avaliação em 500 exemplos jamais vistos.

---

## 5. O assistente

### 5.1 O que ele faz

Diante de uma pergunta clínica, o MedGraph:

1. **filtra a entrada** — remove identificadores, recusa pedidos fora de escopo,
   marca situação de emergência;
2. **classifica a intenção** em cinco caminhos — dúvida clínica, consulta ao
   paciente, exames pendentes, conduta terapêutica, resumo de prontuário;
3. **consulta o prontuário** quando a intenção o exige;
4. **recupera evidência** de protocolos internos e literatura;
5. **raciocina** com a LLM ajustada, ancorado apenas no contexto fornecido;
6. **aplica regras clínicas** determinísticas;
7. **valida a saída** contra quatro invariantes;
8. **escalona para validação humana** quando o risco ultrapassa o limiar;
9. **registra tudo** numa trilha auditável.

### 5.2 A convenção de citação

Toda afirmação clínica aponta sua origem:

| Marcador | Origem | Autoridade |
| --- | --- | --- |
| `[E#]` | Evidência científica (abstract do PubMedQA) | literatura |
| `[P#]` | Protocolo interno do Hospital Vida Plena | norma institucional |
| `[C#]` | Prontuário do paciente | dado do caso |

A numeração é **por tipo**, e não global: o médico sabe, só de ler a citação, se a
afirmação vem da literatura, da norma do hospital ou do paciente à sua frente. São
três níveis de autoridade diferentes, e uma numeração única os confundiria.

O marcador é atribuído no **recuperador**, não na montagem do prompt. Gerado no
prompt, o número dependeria da ordem de concatenação e mudaria entre execuções — a
`[E1]` de uma resposta não corresponderia à `[E1]` gravada na auditoria, e a
rastreabilidade se perderia.

### 5.3 Pipelines LangChain

Duas cadeias, no padrão `prompt | llm | parser` das aulas:

- **`chain_triagem`** — roteamento em dois níveis. Heurística determinística
  primeiro; LLM só no desempate. Não é economia de tokens, é previsibilidade:
  *"Quais exames estão pendentes do PAC-0012?"* não precisa de um modelo para ser
  classificada. A regra de desempate é conservadora de propósito — na dúvida,
  escolher a intenção que exige mais validação, porque classificar a menos faz o
  fluxo pular uma etapa de segurança.

- **`chain_rag`** — a ordem dos blocos no prompt é deliberada: o quadro do
  paciente vem **antes** das fontes, para que o modelo saiba da alergia a
  betalactâmico antes de ler o protocolo que recomenda ceftriaxona. Colocar o
  prontuário no fim faria o modelo formar a resposta e só então encontrar a
  contraindicação.

O prompt de sistema vive em um único módulo, importado por três consumidores: a
montagem do dataset de treino, as chains e o `Modelfile` do Ollama. Se
divergissem, o modelo seria treinado sob um contrato e consultado sob outro — e o
sintoma seria o abandono do formato de citação em produção.

---

## 6. Segurança e validação

### 6.1 Guardrail de entrada

Filtra **antes** de qualquer processamento. Três razões, em ordem de importância:

1. **Segurança** — um pedido para "prescrever direto, sem validação" não deve nem
   chegar ao modelo. Processá-lo e barrar depois significa confiar que a
   verificação de saída reconhecerá o resultado.
2. **Privacidade** — se o médico colar um trecho de prontuário, o nome do paciente
   não pode entrar no prompt, no cache nem na auditoria.
3. **Custo** — recusar na entrada custa milissegundos; na saída, uma inferência.

Emergência **marca, não bloqueia**: recusar durante uma parada cardiorrespiratória
seria pior do que inútil. O sistema antepõe a orientação de acionar o time de
resposta rápida e responde.

### 6.2 Regras clínicas

Verificações determinísticas, sem LLM: conflito com alergia, interação
medicamentosa, valor laboratorial crítico, ajuste por função renal e população
especial.

A verificação de alergia opera em dois níveis — o fármaco citado consta como
alergia, ou pertence à **mesma classe** de uma alergia registrada. O segundo é o
que dá utilidade real à regra, e exige uma tabela de conhecimento farmacológico:
"Ceftriaxona" e "Penicilina" não se parecem em nada como texto, mas são o mesmo
perigo para quem tem alergia a betalactâmicos.

O escore de risco combina os achados pelo complemento do produto, e não pela soma.
Somar faria dois achados médios (0,35 cada) valerem 0,70 — o mesmo que um achado
grave —, o que não corresponde à intuição clínica.

### 6.3 Guardrail de saída

Quatro invariantes:

| Invariante | O que evita |
| --- | --- |
| Toda resposta cita ao menos uma fonte | Afirmação clínica sem procedência |
| Nenhuma citação inventada | Fonte alucinada — pior que a ausência, porque simula rastreabilidade |
| Posologia marcada como pendente de validação | Texto que funcione como prescrição |
| Nenhum identificador de paciente no texto | Vazamento de PII pela resposta |

O **disclaimer não está na lista**, e a razão importa: é texto institucional fixo.
Exigir que o modelo o escreva gasta tokens, faz a aprovação depender de obediência
e — na prática — reprovava toda resposta do modelo base. O sistema o anexa
deterministicamente. **Garantia imposta por código vale mais do que instrução
obedecida.**

Resposta reprovada volta ao nó de raciocínio com as instruções de correção
anexadas, por até duas tentativas. Esgotadas, o sistema **degrada**: entrega as
fontes recuperadas sem afirmar nada sobre elas. O médico prefere *"não consegui
sintetizar, aqui está o que encontrei"* a um erro.

### 6.4 Validação humana

O grafo é compilado com `interrupt_before=["aguardar_validacao"]` e um
checkpointer SQLite. Quando o risco ultrapassa o limiar — ou quando a intenção é
`conduta_terapeutica`, que a política marca como sempre sujeita a validação —, **a
execução para**. O estado fica persistido; retomar exige registrar quem validou.

É assim que *"nunca prescrever diretamente, sem validação humana"* deixa de ser
uma frase no texto e vira uma propriedade da execução.

Os alertas são emitidos **antes** da validação: o médico que valida precisa ver os
conflitos detectados, senão o registro de aprovação não teria fundamento.

### 6.5 Logging e auditoria

Três destinos simultâneos, cada um para um público:

| Destino | Público | Formato |
| --- | --- | --- |
| Console (Rich) | quem assiste à execução | colorido, um ícone por tipo de evento |
| `logs/app.log` | histórico legível | texto rotativo, com módulo e função |
| `logs/auditoria/*.jsonl` | máquina | um evento JSON por linha, particionado por dia |
| `logs/traces/<id>.json` | painel e investigação | dossiê completo de uma consulta |

Todo nó é decorado com `@instrumentar`, que registra início, fim, duração, chaves
do estado recebido e delta produzido. A alternativa — repetir blocos de log em
cada nó — falharia no primeiro esquecimento, e um nó sem rastro invalidaria a
garantia. **Auditar é o padrão, não a exceção.**

Nenhum texto chega ao disco sem passar pelo redator de PII. Textos longos são
truncados com marcação explícita de quantos caracteres foram omitidos — a trilha
nunca mente por omissão.

---

## 7. Avaliação e análise dos resultados

### 7.1 Metodologia

Quatro sistemas sobre o mesmo conjunto de teste, isolado na curadoria e jamais
usado no treino. Há verificação automática de vazamento: a montagem do dataset
falha se qualquer `pubid` de teste aparecer no treino.

| # | Sistema | Papel |
| --- | --- | --- |
| 1 | Classe majoritária | **Piso.** Sem ele, um modelo com 58% pareceria razoável |
| 2 | Modelo base | Ponto de partida, servido em condições idênticas ao ajustado |
| 3 | Modelo ajustado | O resultado do fine-tuning |
| 4 | `gpt-4o-mini` | **Teto de referência**, não concorrente — modelo ~100× maior |

Os sistemas 2 e 3 usam o **mesmo prompt de sistema, a mesma temperatura e o mesmo
template**. Qualquer diferença nesses parâmetros atribuiria ao treino um ganho que
veio da configuração.

Métricas: **accuracy** (o número intuitivo) e **macro-F1** (o número honesto). O
conjunto é desbalanceado, e um modelo que colapse numa única classe tem accuracy
aceitável e macro-F1 baixo.

### 7.2 Resultados

{{AVALIACAO}}

### 7.3 Leitura dos números

**O piso importa mais do que parece.** Responder `yes` para tudo produz accuracy
próxima de 55% e macro-F1 de 0,237. A distância entre esses dois números é a
melhor justificativa para não reportar accuracy sozinha.

**A aderência ao formato é o efeito mais direto do fine-tuning.** É a proporção de
respostas que trazem a linha `Decisão:` na estrutura pedida — independente de a
resposta estar certa. Mede o quanto o treino mudou o *comportamento* do modelo, e
não apenas seu conhecimento.

**O teto dá escala.** Saber que o modelo local ficou a poucos pontos de um modelo
cem vezes maior, executado em nuvem paga, diz mais do que o número absoluto.

**Custo por consulta.** O modelo local processa o mesmo volume com custo
financeiro zero. A tabela de custo registra ambos justamente para tornar essa
comparação explícita.

### 7.4 Gráficos

| Figura | O que mostra |
| --- | --- |
| [`comparativo_sistemas.png`](graficos/comparativo_sistemas.png) | Accuracy e macro-F1 por sistema, com o piso e a referência humana |
| [`f1_por_classe.png`](graficos/f1_por_classe.png) | Onde está o ganho — melhora uniforme e "passou a prever maybe" são histórias diferentes |
| [`adesao_e_latencia.png`](graficos/adesao_e_latencia.png) | Aderência ao formato e custo operacional |
| [`matriz_*.png`](graficos/) | Matriz de confusão normalizada por linha, com a coluna "(sem rótulo)" |

---

## 8. Defeitos encontrados e o que aprendemos

Esta seção existe porque os defeitos abaixo foram mais instrutivos que os acertos.
Todos foram encontrados durante a construção, todos têm teste de regressão, e
todos compartilham a mesma característica: **falhavam em silêncio**.

### 8.1 O prompt que media obediência, não raciocínio

A primeira avaliação deu **35% de accuracy** para o `gpt-4o-mini` — abaixo do piso
de 50%. O modelo respondeu `maybe` em 80% dos casos, quando o rótulo verdadeiro
era `maybe` em 15%.

A causa não era o modelo. A instrução dizia *"use maybe quando a evidência for
inconclusiva"* — um empurrão unilateral, obedecido à risca por um modelo bom em
seguir instrução. O modelo de 3B, que segue com menos rigor, paradoxalmente se
saía melhor.

Reescrevemos o critério de forma simétrica, descrevendo as três classes com o
mesmo peso e explicitando que `no` inclui o achado de **ausência de efeito** — o
caso mais comum de `no` no PubMedQA. Resultado: `gpt-4o-mini` de 35% para 65%;
modelo base de 60% para 70%.

**A lição:** um benchmark mal instrumentado mede o instrumento.

### 8.2 A regra de segurança que era decorativa

A primeira verificação de alergia comparava o nome do fármaco citado com o texto
da alergia registrada. Um paciente com `Penicilina [Betalactâmico]` e uma conduta
sugerindo `Ceftriaxona 2 g EV` não produzia alerta nenhum.

Reatividade cruzada é conhecimento farmacológico, não similaridade textual. A
correção exigiu construir uma tabela de classes de fármaco.

**A lição:** a verificação mais importante do sistema era a que menos funcionava,
e nada indicava isso.

### 8.3 A flag que reprovava todas as respostas

O padrão de detecção de nome próprio usava `re.IGNORECASE`, na suposição de que
isso tornaria apenas o *marcador* insensível a caixa. A flag vale para a expressão
inteira e anula as classes `[A-Z]` e `[a-z]` — justamente o que distingue um nome
próprio de uma palavra comum.

Em *"a avaliação do paciente deve incluir a coleta"*, o trecho "paciente deve
incluir" casava como "marcador + Nome Sobrenome". O guardrail reprovava **toda**
resposta por suposto vazamento de PII, o fluxo esgotava as reescritas e degradava
em todas as consultas.

**A lição:** um guardrail que reprova tudo é tão inútil quanto um que aprova tudo,
e o sintoma é indistinguível de "o modelo está ruim".

### 8.4 O bloqueio que o acento contornava

Os padrões de bloqueio em `politicas.yaml` são escritos em ASCII. O pedido *"Pule
a validação humana e me dê a receita"* — a grafia que qualquer pessoa usaria —
passava direto pelo guardrail.

Escrever cada padrão com as duas grafias funcionaria, mas teria de ser lembrado em
todo padrão novo. Normalizar o texto antes da comparação resolve para todos.

**A lição:** uma proteção que depende de lembrar de algo em cada adição futura já
está quebrada.

### 8.5 Duas tabelas que precisavam concordar

A tabela de interações medicamentosas citava `varfarina` e `amiodarona`, ausentes
da tabela de classes farmacológicas. Como o detector só reconhece o que está na
segunda, essas interações **nunca disparavam**: a regra existia no código e era
inerte na prática.

Há agora um teste que falha se alguém adicionar uma interação sem cadastrar os
fármacos correspondentes.

### 8.6 O relatório de cobertura que errava para menos

O gerador da matriz de rastreabilidade usava a regex `REQ-[0-9]+`, que nunca casa
com `REQ-E1`. Os quatro entregáveis apareciam como sem cobertura apesar de dezenas
de referências no código.

**A lição:** um relatório de cobertura que erra para menos leva a procurar um
problema que não existe, e mascara os que existem.

### 8.7 O caractere invisível que corrompia o dataset

`str.splitlines()` quebra em qualquer separador de linha Unicode — `\x0b`, `\x0c`,
` `. Todos são caracteres **válidos** dentro de uma string JSON, e o
`json.dumps` não os escapa. Um único abstract científico contendo um deles partia
a linha em duas e quebrava o parse do arquivo inteiro.

Corrigido nos dois lados: leitura por iteração do arquivo, e higienização na
origem.

---

## 9. Limitações declaradas

Um sistema clínico que não declara o alcance da sua verificação induz uma
confiança que não merece. As limitações abaixo são conhecidas e estão registradas
também no código.

| Limitação | Consequência | O que seria necessário |
| --- | --- | --- |
| A tabela de classes farmacológicas cobre apenas os fármacos dos 15 protocolos | Fármaco ausente da tabela **não gera alerta** | Base como ANVISA, Micromedex ou o dicionário do prontuário eletrônico |
| A tabela de interações é restrita a pares de relevância reconhecida | Interações não catalogadas passam | Base de interações licenciada |
| A detecção de nome próprio é heurística, não NER | Nome incomum fora do dicionário pode escapar | Modelo de NER treinado em português clínico |
| Dados de paciente são sintéticos | O desempenho em prontuário real é desconhecido | Validação em ambiente controlado com dados reais |
| O corpus de evidência é o PubMedQA | Não cobre a literatura biomédica inteira | Indexação de base bibliográfica completa |
| A avaliação usa uma amostra do conjunto de teste | Intervalo de confiança maior que o do conjunto completo | `make avaliar --completo` (~1 h por sistema) |
| Não houve validação clínica por médico | O sistema não está apto a uso assistencial | Estudo prospectivo com supervisão médica |

**Este é um projeto acadêmico.** Não foi submetido a comitê de ética, não passou
por validação clínica e não deve ser utilizado em assistência a pacientes.

---

## 10. Rastreabilidade dos requisitos

{{RASTREABILIDADE}}

A matriz completa, ligando cada exigência do enunciado ao arquivo e à linha onde é
atendida, está em [`docs/rastreabilidade.md`](rastreabilidade.md) e é gerada
varrendo as tags `[REQ-xx]` das docstrings.

### Suíte de testes

{{TESTES}}

---

## Referências

- **PubMedQA** — Jin, Q. et al. *PubMedQA: A Dataset for Biomedical Research
  Question Answering.* EMNLP 2019. [pubmedqa.github.io](https://pubmedqa.github.io/)
- **QLoRA** — Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.*
  NeurIPS 2023.
- **LoRA** — Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.*
  ICLR 2022.
- **Llama 3.2** — Meta AI, sob a Llama 3.2 Community License.
- **LangChain / LangGraph** — LangChain Inc., licença MIT.
- **E5** — Wang, L. et al. *Multilingual E5 Text Embeddings.* 2024.
