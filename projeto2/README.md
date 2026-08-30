# MedGraph Lite

**Tech Challenge · Fase 3 · Pós-Tech 8IADT** — Hospital Vida Plena (cenário fictício)

Assistente clínico com LLM ajustada por fine-tuning, orquestrado com LangChain e
LangGraph, com limites de atuação, trilha de auditoria e citação de fonte em toda
resposta.

**Roda inteiro em um notebook do Colab, em cerca de 25 minutos.**

---

## Abrir e executar

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/projeto2/notebooks/medgraph_lite.ipynb
```

1. **Ambiente de execução → Alterar o tipo → T4 GPU**
2. Rode a seção 1 (verifica a GPU) e a 2 (instala, ~3 min)
3. **Reinicie a sessão** e continue da seção 3
4. Rode até o fim

Não precisa de conta no Hugging Face, de aceite de licença, nem de chave de API.
O modelo base é aberto.

## O que acontece, seção a seção

| Seção | O que faz | Tempo |
| --- | --- | ---: |
| 1-2 | GPU e dependências | 3 min |
| 3 | PubMedQA, protocolos do hospital, anonimização | 1 min |
| 4 | Base de prontuários em SQLite | instantâneo |
| 5 | **Fine-tuning QLoRA** e avaliação antes do ajuste | 10 min |
| 6 | Avaliação depois, e o gráfico antes × depois | 2 min |
| 7 | Índice FAISS de evidência | 1 min |
| 8 | Assistente LangChain com prontuário e fontes | 1 min |
| 9 | **Fluxo LangGraph**: 4 consultas, 4 caminhos | 2 min |
| 10 | Conclusão e limitações | — |

## Os quatro gráficos

Gerados em matplotlib, dentro do próprio notebook:

1. **Curva de perda** — o treino funcionou?
2. **Antes × depois** — o que o ajuste melhorou, em adesão ao formato e acurácia
3. **Caminhos no grafo** — por quais nós cada consulta passou, com a latência de cada um
4. **Achados por severidade** — quantos alertas cada nível de gravidade produziu

## Estrutura

```
projeto2/
├── medgraph_lite/
│   ├── dados.py        PubMedQA, protocolos sintéticos, anonimização
│   ├── prontuario.py   base SQLite e o modelo de paciente
│   ├── treino.py       configuração do QLoRA e formato do prompt
│   ├── rag.py          índice FAISS e recuperação com marcador de fonte
│   ├── guardrails.py   limites de atuação e regras clínicas
│   ├── grafo.py        fluxo LangGraph de seis nós
│   └── graficos.py     as quatro figuras
├── notebooks/
│   └── medgraph_lite.ipynb
└── requirements.txt
```

Cerca de 950 linhas de Python, comentadas.

## O fluxo

```
guardrail_entrada --(recusado)--> montar_resposta
        |
consultar_prontuario -> recuperar_evidencia -> responder -> verificar_resposta
        |                                                          |
        |                              (crítico) --> validacao_humana
        |                                                          |
        +------------------------------------------> montar_resposta
```

Cada nó registra o que fez, quanto tempo levou e o que decidiu. Essa trilha é o
logging que o item 3 do enunciado pede — e é o que o gráfico de caminhos desenha.

## O princípio

> O assistente **nunca prescreve**. Ele apresenta evidência, aponta a fonte de cada
> afirmação e devolve a decisão ao médico responsável.

Quando a resposta conflita com uma alergia registrada, quando há interação
medicamentosa, ou quando falta citação de fonte, a execução **para** e aguarda
validação humana. Isso está em código verificável, não no prompt.

## Decisões técnicas

| Decisão | Por quê |
| --- | --- |
| **Qwen2.5-0.5B** como base | Aberto (sem licença a esperar) e treina em ~8 min. O enunciado deixa a escolha livre |
| **QLoRA em 4 bits** | Treina ~1% dos parâmetros; cabe folgado na T4 gratuita |
| **Modelo em memória**, sem GGUF nem Ollama | Elimina 25 minutos de exportação e três dependências externas que quebram com atualização de versão |
| **FAISS local** | Mesmo vector store das aulas, sem serviço externo |
| **Fármacos por classe**, não por nome | "Penicilina" não casa com "Ceftriaxona" por texto — mas as duas são betalactâmicos |
| **Evitação por frase**, não por proximidade | "Evitar penicilina. Iniciar ceftriaxona." — uma janela de caracteres classificaria a ceftriaxona como evitada |

## Requisitos do enunciado

| Requisito | Onde |
| --- | --- |
| Fine-tuning com protocolos, FAQ e modelos de documento | `dados.py`, `treino.py`, seção 5 |
| Preprocessing, anonimização e curadoria | `dados.anonimizar`, seção 3.1 |
| Pipeline LangChain com a LLM customizada | seção 8 |
| Consulta a base estruturada | `prontuario.py`, seção 4 |
| Contextualização com dados do paciente | `grafo.no_responder`, seção 9 |
| Limites de atuação | `guardrails.py`, seção 9 |
| Logging para auditoria | `grafo._registrar`, seção 9 |
| Explainability por fonte | `rag.py` + `guardrails.verificar_resposta` |
| Código modularizado | pacote `medgraph_lite/` |
| Fluxos LangGraph | `grafo.py` |
| Dataset sintético/anonimizado | `dados.PROTOCOLOS`, `dados.FAQ` |

## Transparência

Nenhum dado real de paciente é utilizado. Protocolos, FAQ e prontuários do Hospital
Vida Plena são sintéticos. O pipeline de anonimização é aplicado mesmo assim,
demonstrando a técnica e garantindo que o mesmo código funcionaria sobre dados reais.

Projeto acadêmico. Sem validação clínica, não deve ser usado em assistência a
pacientes.
