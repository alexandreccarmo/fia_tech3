# Guia — executar o fine-tuning no Google Colab

**MedGraph · Tech Challenge Fase 3 · 8IADT**

Esta é a única etapa do projeto que não roda no MacBook: ajustar um modelo de 3
bilhões de parâmetros exige GPU com CUDA. Tempo total: **60 a 90 minutos**, quase
todo de espera.

---

## Antes de abrir o Colab — 5 minutos

Faça estes dois passos primeiro. Se deixar para depois, o notebook vai parar no
meio esperando por eles.

### 1. Aceitar a licença do Llama 3.2

O modelo é *gated*: exige aceite, gratuito e com aprovação imediata.

1. Crie uma conta em [huggingface.co/join](https://huggingface.co/join) se ainda não tiver
2. Acesse **[huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)**
3. Preencha o formulário e clique em **Submit**
4. Aguarde o selo mudar para **"You have been granted access to this model"** — costuma ser imediato

> **Não quer lidar com o *gating*?** Há uma alternativa sem licença. Na célula de
> código 7 do notebook, troque:
> ```python
> MODELO_BASE = "Qwen/Qwen2.5-3B-Instruct"
> ```
> O Qwen2.5-3B é aberto, de tamanho equivalente, e tem suporte a português
> comparável. O restante do notebook não muda.

### 2. Criar o token do Hugging Face

1. Vá em **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**
2. **Create new token** → aba **Write**
3. Nome: `colab-medgraph`
4. **Create token** e copie o valor (`hf_...`) — ele só aparece uma vez

O token precisa ser de **escrita** porque o segundo notebook publica o modelo
convertido no Hub.

---

## Abrir o notebook no Colab

Como o repositório é público, o Colab abre direto do GitHub — sem baixar nem
subir arquivo:

**[▶ Abrir 01_finetune_qlora_pubmedqa.ipynb no Colab](https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/colab/01_finetune_qlora_pubmedqa.ipynb)**

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/colab/01_finetune_qlora_pubmedqa.ipynb
```

*Alternativa:* dentro do Colab, **Arquivo → Abrir notebook → GitHub**, cole
`alexandreccarmo/fia_tech3` e escolha o notebook na lista.

### Ativar a GPU — não pule este passo

**Ambiente de execução → Alterar o tipo de ambiente de execução → Acelerador de
hardware: `T4 GPU` → Salvar**

A T4 gratuita basta. O notebook falha logo na primeira célula se não houver GPU —
de propósito: descobrir isso em segundos é melhor do que depois de vinte minutos
baixando o modelo.

---

## Executar — o que esperar em cada bloco

| Célula | O que faz | Tempo | Atenção |
| --- | --- | --- | --- |
| 1 | Mostra a GPU disponível | instantâneo | Deve aparecer "Tesla T4" |
| 2 | Instala as bibliotecas | ~3 min | **Reinicie a sessão depois** ⚠️ |
| 3 | Clona o repositório | ~30 s | Traz o dataset junto |
| 4 | Confere GPU e versões | instantâneo | Avisa que a T4 não tem bfloat16 — é esperado |
| 5 | Login no Hugging Face | ~30 s | Cole o token quando pedir |
| 6 | Carrega o dataset | ~20 s | Mostra um exemplo completo |
| 7 | Baixa o modelo em 4 bits | ~5 min | ~2 GB |
| 8 | Configura o LoRA | instantâneo | Mostra quantos parâmetros treinam |
| 9 | Monta o treinador | instantâneo | Informa quantos passos serão dados |
| **10** | **Treina** | **~50 min** | **Mantenha a aba aberta** |
| 11 | Desenha a curva de perda | instantâneo | Validação deve cair junto com o treino |
| 12 | Testa o formato em 5 exemplos | ~1 min | Espera-se 4 ou 5 acertos de formato |
| 13 | Grava o adapter | ~10 s | ~50 MB |
| 14 | Baixa o `.zip` | ~20 s | Guarde este arquivo |

### ⚠️ O ponto onde todo mundo tropeça

Depois da **célula 2**, o Colab precisa reiniciar para carregar as bibliotecas
novas. O notebook avisa em letras maiúsculas.

**Ambiente de execução → Reiniciar sessão** — e então continue **da célula 3**,
não do começo. Reexecutar a célula 2 depois do restart só perde tempo.

### Durante o treino

O Colab gratuito **desconecta sessões ociosas**. Deixe a aba aberta e visível.
Se cair, não recomece do zero: há checkpoints a cada 100 passos, e basta rodar

```python
treinador.train(resume_from_checkpoint=True)
```

---

## Depois do treino — o segundo notebook

**[▶ Abrir 02_exportar_gguf.ipynb no Colab](https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/colab/02_exportar_gguf.ipynb)**

Ele funde o adapter ao modelo base, converte para GGUF quantizado e publica no
Hugging Face Hub. Cerca de **25 minutos**.

Se rodar na **mesma sessão** do notebook 1, o adapter já está em disco. Em sessão
nova, ele pede o upload do `medgraph-adapter.zip` que você baixou.

Ao final, o notebook imprime o nome exato do repositório criado. **Anote-o.**

---

## De volta à sua máquina

```bash
cd ~/Desktop/FIAP/projeto/tech_challenge_fiap3/fia_tech3
```

1. **Descompacte o adapter** (ele é versionado no Git — é o artefato do treino):

   ```bash
   unzip ~/Downloads/medgraph-adapter.zip -d models/adapters/
   ```

2. **Ajuste o `.env`** com o repositório que o notebook 2 imprimiu:

   ```
   REPO_GGUF_HF=seu-usuario/medgraph-llama32-3b-gguf
   HF_TOKEN=hf_...
   OLLAMA_MODEL=medgraph
   ```

3. **Registre o modelo ajustado no Ollama:**

   ```bash
   make modelo -- --ajustado
   ```

4. **Reavalie** — a tabela ganha a coluna do modelo ajustado:

   ```bash
   make avaliar
   make relatorio
   ```

---

## Se algo der errado

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| `Nenhuma GPU disponível` | Runtime sem acelerador | Ambiente de execução → T4 GPU |
| `401` ao carregar o modelo | Licença do Llama não aceita | Aceite em huggingface.co e refaça o login |
| `ModuleNotFoundError` na célula 4 | Sessão não foi reiniciada | Reinicie e continue da célula 3 |
| `CUDA out of memory` | Sequência longa demais | Antes da célula 9: `colab_utils.CONFIG_PADRAO["max_seq_length"] = 768` |
| Sessão desconectou | Ociosidade do Colab gratuito | `treinador.train(resume_from_checkpoint=True)` |
| `TypeError` no `SFTTrainer` | Mudança de API do `trl` | Leia a lista de argumentos ignorados que a célula 9 imprime |
| Formato incorreto na célula 12 | Treino insuficiente | Aumente `num_train_epochs` para 3 e retreine |

---

## Quanto custa

**Nada.** A T4 do Colab gratuito é suficiente, o Hugging Face é gratuito, e o
modelo resultante roda localmente sem custo por consulta.

A única limitação do plano gratuito é a cota diária de GPU: se você esgotar,
espere algumas horas ou use outra conta Google.
