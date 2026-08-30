# Guia — executar o fine-tuning no Google Colab

**MedGraph · Tech Challenge Fase 3 · 8IADT**

Esta é a única etapa do projeto que não roda no MacBook: ajustar um modelo de 3
bilhões de parâmetros exige GPU com CUDA.

**Reserve de 6 a 7 horas** para a rodada completa na T4 gratuita — quase tudo de
espera. Se isso não couber no seu dia, leia *Treinar em menos tempo* mais abaixo
antes de começar: há uma receita de teste rápido e as opções para a rodada de
entrega.

---

## Antes de abrir o Colab — 5 minutos

Faça estes dois passos primeiro. Se deixar para depois, o notebook vai parar no
meio esperando por eles.

### 1. Aceitar a licença do Llama 3.2

O modelo é *gated*: exige aceite, gratuito e com aprovação imediata.

1. Crie uma conta em [huggingface.co/join](https://huggingface.co/join) se ainda não tiver
2. Acesse **[huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)**
3. Preencha o formulário e clique em **Submit**
4. Aguarde o selo mudar para **"You have been granted access to this model"**

> ⏳ **A aprovação NÃO é garantida nem instantânea.** Depois de enviar, o mais
> comum é ver:
>
> > *"Your request to access this repository has been submitted and is awaiting a
> > review from the repository authors."*
>
> Isso é a fila da Meta, não um erro seu. Costuma sair rápido, mas pode levar de
> minutos a horas — e não há como consultar a posição.
>
> **Não espere com a sessão do Colab aberta.** Uma sessão com GPU alocada consome
> a cota diária do plano gratuito mesmo sem estar treinando. Use o Qwen (abaixo)
> e siga adiante — o pedido continua na fila sozinho, e você não fica parado.
>
> **Mas entenda o que isso custa.** Um adapter LoRA são matrizes com as dimensões
> da arquitetura em que foi treinado: o do Qwen **não serve** para o Llama. Se o
> aceite sair e você quiser entregar com Llama, o treino é refeito do zero
> (~5 h 40) e a exportação também (~25 min). O que você ganha rodando o Qwen é
> validar o pipeline inteiro antes; o modelo em si é descartável.

> **Se o aceite não sair, ou você não quiser esperar:** a célula 7 traz a
> alternativa já pronta, comentada. Basta trocar qual das duas linhas está ativa:
>
> ```python
> # MODELO_BASE = "meta-llama/Llama-3.2-3B-Instruct"
> MODELO_BASE = "Qwen/Qwen2.5-3B-Instruct"
> ```
>
> O Qwen2.5-3B é aberto, do mesmo tamanho e com suporte a português comparável.
> A troca é segura: os módulos-alvo do LoRA são idênticos nas duas arquiteturas e
> o Modelfile do Ollama não fixa template. **Use o mesmo valor no notebook 02** —
> senão ele funde o adapter no modelo errado.
>
> O enunciado pede *"um modelo LLM (como LLaMA, Falcon ou um outro)"* — a escolha
> é livre, e a troca não afeta a avaliação do trabalho.

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
| 5 | Login no Hugging Face | ~1 min | **Autorização por código** — veja abaixo |
| 6 | Carrega o dataset | ~20 s | Mostra um exemplo completo |
| 7 | Baixa os pesos e quantiza | 3–8 min | Baixa ~6 GB em fp16; a quantização para 4 bits é em memória |
| 8 | Configura o LoRA | instantâneo | Mostra quantos parâmetros treinam |
| 9 | Monta o treinador | instantâneo | Informa quantos passos serão dados |
| **10** | **Treina** | **~5 h 40** | Medido: 42 s/passo × 484 passos numa T4. Veja *Treinar em menos tempo* |
| 11 | Desenha a curva de perda | instantâneo | Validação deve cair junto com o treino |
| 12 | Testa o formato em 5 exemplos | ~1 min | Espera-se 4 ou 5 acertos de formato |
| 13 | Grava o adapter | ~10 s | ~50 MB |
| 14 | Baixa o `.zip` | ~20 s | Guarde este arquivo |

### Como saber se a célula 7 travou ou só está lenta

Ela mostra barras de progresso dos arquivos `.safetensors`. Se estiverem
avançando, está tudo bem — a rede do Colab varia.

Ao terminar, imprime (valores medidos com o Llama-3.2-3B-Instruct):

```
Parâmetros: 3.21 B
VRAM ocupada: 2.09 GB
```

A contagem de parâmetros identifica qual modelo carregou de fato: **3,21 B é o
Llama-3.2-3B**, e **3,09 B é o Qwen2.5-3B**. Se o número não for o do modelo que
você escolheu na célula 7, pare aqui — o resto da execução seria sobre a
arquitetura errada.

O `VRAM ocupada` bem abaixo dos ~6 GB baixados é a confirmação de que a
quantização em 4 bits funcionou. Se ocupasse os 6 GB inteiros, o modelo teria
carregado em precisão cheia — e o treino não caberia na T4.

### A célula 5 não pede o token — pede um código

As versões recentes do `huggingface_hub` abandonaram o campo de texto para o
token e usam **autorização por dispositivo**. A célula imprime algo assim:

```
To log in, open this URL and enter the code:
https://hf.co/oauth/device

    AOAH-6YNV

Waiting for authorization...
```

1. Abra **[hf.co/oauth/device](https://hf.co/oauth/device)** em outra aba
   (é preciso estar logado no huggingface.co nela)
2. Digite o código **com o hífen** — os espaços na tela são só visuais
3. Clique em **Authorize**
4. Volte ao Colab: a célula termina sozinha e imprime `Autenticado como: ...`

O código **expira em poucos minutos**. Se passar do prazo, reexecute a célula 5
para gerar outro.

> Neste fluxo o token que você criou **não é usado aqui** — a autenticação vem da
> sessão do navegador. Guarde-o mesmo assim: ele vai para o `.env` na sua máquina
> ao final, e o segundo notebook pode pedi-lo para publicar no Hub.

### ⚠️ O ponto onde todo mundo tropeça

Depois da **célula 2**, o Colab precisa reiniciar para carregar as bibliotecas
novas. O notebook avisa em letras maiúsculas.

**Ambiente de execução → Reiniciar sessão** — e então continue **da célula 3**,
não do começo. Reexecutar a célula 2 depois do restart só perde tempo.

### Durante o treino

O Colab gratuito **desconecta sessões ociosas**. Deixe a aba aberta e visível.

**Se a sessão cair**, o kernel morre e a variável `treinador` deixa de existir —
não dá para rodar o comando de retomada direto. O procedimento é:

**1.** Reconecte e verifique o que sobreviveu:

```python
!ls -la /content/saida_treino/
```

**2.** Interprete o resultado:

| O que aparece | Situação | O que fazer |
| --- | --- | --- |
| Pastas `checkpoint-100`, `checkpoint-200`… | A VM sobreviveu, só o kernel caiu | Siga para o passo 3 |
| `No such file or directory` | A VM foi reciclada e o disco apagado | Recomece da célula 2 |

**3.** Reconstrua o treinador rodando as **células 3 a 9** — e a 2 apenas se der
`ModuleNotFoundError`, o que só acontece se a VM foi trocada.

**4.** Na **célula 10**, troque para:

```python
resultado = treinador.train(resume_from_checkpoint=True)
```

Ele retoma do checkpoint mais recente. Parado no passo 300 de 484, faltam ~2 h em
vez de 5 h 40.

### Treinar em menos tempo

**O número medido:** numa T4 do Colab gratuito, com Qwen2.5-3B e a configuração
padrão do projeto, o treino roda a **42 segundos por passo**. São 484 passos (3.871
exemplos, 2 épocas, lote efetivo 16), ou seja **~5 h 40 min**.

Isso é o comportamento normal do QLoRA em 4 bits numa T4, e não um defeito: cada
multiplicação de matriz precisa desquantizar os pesos antes de calcular. Não
adianta procurar o que está errado — não há.

A consequência prática é séria: **a rodada completa não cabe na cota diária do
Colab gratuito**, que costuma ficar entre 3 e 4 horas de GPU.

#### Teste rápido — validar o pipeline em ~20 minutos

Serve para confirmar que tudo roda de ponta a ponta com um modelo base novo, sem
gastar a tarde. Cole numa célula **entre a 6 e a 7** — ela precisa de `dados`, que
a célula 6 cria:

```python
# Treino curto: valida o pipeline, NAO produz o modelo da entrega.
ALVO_MIN       = 20
SEG_POR_PASSO  = 45     # medido: 42 s/passo numa T4; margem para o Llama, um pouco maior
LOTE_EFETIVO   = (colab_utils.CONFIG_PADRAO["per_device_train_batch_size"]
                  * colab_utils.CONFIG_PADRAO["gradient_accumulation_steps"])

passos   = int(ALVO_MIN * 60 / SEG_POR_PASSO)
exemplos = passos * LOTE_EFETIVO

colab_utils.CONFIG_PADRAO["num_train_epochs"] = 1
# A escala dos intervalos acompanha o treino. Com os valores padrao (eval a cada
# 100 passos) um treino de 26 passos nao avaliaria nenhuma vez, e a curva de
# perda da celula 11 sairia sem a linha de validacao.
colab_utils.CONFIG_PADRAO["logging_steps"]    = 2
colab_utils.CONFIG_PADRAO["eval_steps"]       = max(1, passos // 4)
colab_utils.CONFIG_PADRAO["save_steps"]       = passos
colab_utils.CONFIG_PADRAO["save_total_limit"] = 1

dados["train"]      = dados["train"].shuffle(seed=42).select(range(min(exemplos, len(dados["train"]))))
dados["validation"] = dados["validation"].select(range(min(64, len(dados["validation"]))))

print(f"Treino curto: {len(dados['train'])} exemplos, ~{passos} passos, ~{ALVO_MIN} min")
```

Com ~26 passos o modelo mal começa a aprender. **Não espere 4 ou 5 na célula 12** —
o objetivo aqui é ver o pipeline funcionar, não o modelo ficar bom. Julgar a
qualidade por este treino levaria à conclusão errada.

#### Para a rodada de entrega

| Caminho | Tempo | O que custa |
| --- | ---: | --- |
| Padrão (2 épocas, 3.871 ex.) | ~5 h 40 | Não cabe na cota gratuita |
| 1 época | ~2 h 50 | Menos convergência; cabe numa sessão longa |
| 1 época + 2.000 exemplos | ~1 h 30 | Dataset menor — declarar no relatório |
| Duas sessões com `USAR_DRIVE = True` | 2 × ~3 h | Nada na qualidade; exige retomar do checkpoint |
| Colab Pro (L4 ou A100) | ~30–60 min | Assinatura mensal |

Há ainda uma alavanca de velocidade sem perda de dados: desligar o
`use_gradient_checkpointing` na célula 8 acelera de 30 a 40%, ao custo de bem mais
VRAM — na T4 isso pode estourar, e o estouro vem no meio do treino.

Não reduza o `max_seq_length` para ganhar tempo: os exemplos têm ~900 tokens em
média, e cortar para 512 truncaria a resposta esperada na metade deles. O modelo
aprenderia a responder pela metade.

### Trocar de modelo base depois de já ter treinado

Se você treinou com o Qwen e o aceite do Llama saiu, **não basta trocar a linha
da célula 7 e continuar dali.** Duas coisas atrapalham.

**A VRAM.** A célula 7 carrega o novo modelo com o antigo ainda na GPU — Python
avalia `from_pretrained(...)` antes de reatribuir `modelo`. E o antigo não é
liberado nem depois, porque `treinador.model` continua apontando para ele.

Em 4 bits cada modelo ocupa ~2,2 GB, então os dois provavelmente **cabem** na T4
e o carregamento passa. O problema aparece depois: o treino é o que consome
memória de verdade — ativações, gradientes e cache —, e ele passa a disputar a
GPU com um modelo inteiro que ninguém mais usa. Um `CUDA out of memory` na
célula 10, aos vinte minutos de treino, custa muito mais caro do que um erro no
carregamento.

**Os detritos do treino anterior.** Os `checkpoint-N` em `/content/saida_treino`
são da outra arquitetura. Eles somem aos poucos (`save_total_limit=2`), mas se a
sessão cair no meio e você usar `resume_from_checkpoint=True`, o Trainer pode
pegar um checkpoint do modelo antigo. O adapter em `models/adapters/` tem o
mesmo problema.

O procedimento seguro:

**1.** Se quiser guardar o adapter anterior para comparar, baixe o zip da célula
14 agora — o passo seguinte o apaga.

**2. Ambiente de execução → Reiniciar sessão.** Isso libera a GPU de forma
determinística, sem depender de `del` e coleta de lixo.

**3.** Numa célula, limpe o que sobrou em disco. **Use caminhos absolutos:**

```python
!rm -rf /content/saida_treino /content/fia_tech3
!ls -la /content/
```

O `rm -rf` não reclama de caminho inexistente — sem o `ls` depois, ele é mudo e
você não tem como saber se apagou algo. E o caminho precisa ser absoluto porque
a célula 3 faz `os.chdir("/content/fia_tech3")`: reiniciada a sessão, o diretório
de trabalho volta a ser `/content`, e um `models/adapters/...` relativo apontaria
para uma pasta que não existe. O comando sairia silencioso sem apagar nada, e o
adapter antigo continuaria lá.

Apagar o clone inteiro, e não só o adapter, tem um segundo motivo: a célula 3 só
clona **se a pasta não existir**. Sem isso você seguiria com o código da sessão
anterior, sem as correções publicadas desde então.

**4.** Troque a linha ativa da célula 7 e rode **da célula 3 em diante** — não da
2, os pacotes continuam instalados no disco da VM. O login da célula 5 precisa
ser refeito, com um código novo.

**5.** Use o mesmo modelo no notebook 02.

Reserve ~75 min de GPU para a rodada completa (treino + exportação) e confira
quanto de cota você já gastou no dia.

### Proteger o treino contra a reciclagem da VM

`/content` é o disco **efêmero** do Colab. Sobrevive a uma desconexão, mas não à
reciclagem da máquina — que acontece em desconexões longas ou ao esgotar a cota.

A célula 9 traz a opção de gravar no Google Drive:

```python
USAR_DRIVE = True
```

Custa cerca de um minuto de configuração e ~1,5 GB do seu Drive (dois
checkpoints). Vale a pena se você estiver treinando em horário de pico ou já
tiver perdido uma sessão hoje.

---

## O que sobrevive quando você fecha o Colab

Três coisas diferentes, com destinos diferentes.

**As variáveis não sobrevivem — nunca.** `modelo`, `treinador`, `dados`: ao
reconectar, o kernel é outro. Isso não é problema para *visualizar* o resultado,
só para continuar executando.

**Os arquivos em `/content` são efêmeros.** Sobrevivem a uma desconexão curta,
mas somem quando a VM é reciclada — junto com o repositório clonado e o adapter.

**As saídas das células só sobrevivem se você salvar o notebook.** Aberto pelo
link do GitHub, o Colab trata a sessão como descartável: fechar a aba sem salvar
descarta a curva de perda, a saída da célula 12 e todo o resto.

Para poder reabrir e **ver** o resultado sem executar nada, depois que o treino
terminar:

**Arquivo → Salvar uma cópia no Drive**

A cópia guarda o notebook com todas as saídas. Reabrindo-a, tudo aparece sem
rodar uma célula. `Arquivo → Fazer download → .ipynb` tem o mesmo efeito, no seu
computador.

> ⚠️ **Não use "Salvar uma cópia no GitHub"** se você trocou o `MODELO_BASE` para
> o Qwen. Isso levaria a edição para o repositório, que ficaria dizendo que o
> projeto usa um modelo diferente do que o relatório afirma. Há um teste que
> acusa isso, mas é melhor não criar o problema.

**O que realmente precisa ser preservado, porém, não é a visualização** — é o
`.zip` da célula 14. Ele contém o adapter, o tokenizador, o `cartao_de_treino.json`
(com o histórico completo de perdas) e a `curva_de_perda.png`. Com esse arquivo
na sua máquina, o Colab pode ser descartado inteiro.

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

   Vêm junto o `cartao_de_treino.json`, com a procedência do treino, e a
   `curva_de_perda.png` — os dois artefatos que o roteiro do vídeo manda mostrar
   no Bloco 3. Para deixar o gráfico junto dos demais:

   ```bash
   cp models/adapters/medgraph-llama32-3b-lora/curva_de_perda.png docs/graficos/
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
| `Nenhuma GPU disponível` | Runtime sem acelerador **ou cota diária esgotada** | Rode `!nvidia-smi`: se o comando não existir e o menu já mostrar T4 GPU, é cota — ver *Quando a cota de GPU acaba* |
| `GatedRepoError` ou `403` na célula 7 | Conta sem acesso ao Llama — pedido não enviado, pendente, ou feito por outra conta | Troque para o Qwen na própria célula 7 (a linha já está lá, comentada) e reexecute a partir dela |
| Célula 5 travada em `Waiting for authorization` | Código expirou ou não foi autorizado | Reexecute a célula 5 e autorize o código novo |
| `ModuleNotFoundError` na célula 4 | Sessão não foi reiniciada | Reinicie e continue da célula 3 |
| `NotImplementedError: ... not implemented for 'BFloat16'` na célula 10 | Adaptadores LoRA em bf16 com treino em fp16 — o modelo carregou na precisão do próprio `config.json` | Corrigido automaticamente pelo `montar_treinador`. Se a célula 9 não imprimir "parâmetro(s) treinável(is) convertido(s) para float32", seu notebook é anterior à correção: reabra do GitHub |
| `CUDA out of memory` | Sequência longa demais | Antes da célula 9: `colab_utils.CONFIG_PADRAO["max_seq_length"] = 768` |
| Sessão desconectou | Ociosidade do Colab gratuito | `treinador.train(resume_from_checkpoint=True)` |
| `TypeError` no `SFTTrainer` | Mudança de API do `trl` | Leia a lista de argumentos ignorados que a célula 9 imprime |
| Formato incorreto na célula 12 | Treino insuficiente | Aumente `num_train_epochs` para 3 e retreine |

---

## Quando a cota de GPU acaba

O plano gratuito dá algo entre 3 e 4 horas de GPU por dia, e o treino completo
consome ~5 h 40 — então esbarrar na cota é o caso comum, não a exceção.

O que torna isso confuso é que **o Colab não recusa a conexão.** Ele conecta em
CPU, mantém "T4 GPU" selecionada no menu, e o notebook só falha quando alguma
célula procura a GPU. Você olha o menu, vê a T4 marcada, e conclui que o problema
é outro.

Para confirmar em dois segundos:

```python
!nvidia-smi
```

Se o comando não existir, não há GPU no ambiente. Com o menu já em T4 GPU, isso é
cota.

**Não há o que consertar no notebook.** A cota volta sozinha, tipicamente em
algumas horas, e não há como consultar quanto falta nem a posição na fila. As
saídas são esperar, usar outra conta Google, ou assinar o Colab Pro — que além de
liberar a cota costuma dar L4 ou A100, onde o mesmo treino cai para 30-60 min.

Para gastar menos cota por rodada, veja *Treinar em menos tempo*.

---

## Quanto custa

**Nada.** A T4 do Colab gratuito é suficiente, o Hugging Face é gratuito, e o
modelo resultante roda localmente sem custo por consulta.

A única limitação do plano gratuito é a cota diária de GPU: se você esgotar,
espere algumas horas ou use outra conta Google.
