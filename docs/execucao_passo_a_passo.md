# Execução passo a passo — do zero ao projeto concluído

**MedGraph · Tech Challenge Fase 3 · 8IADT**

Este documento descreve **toda** a execução do projeto, célula a célula, do
primeiro clone até o vídeo de entrega. Ele existe para ser seguido sem
conhecimento prévio do projeto e para servir de base a quem precise explicar
o procedimento a outra pessoa.

Os tempos citados foram **medidos** numa T4 do Colab gratuito com
Llama-3.2-3B-Instruct, e não estimados. Onde um número for aproximado, está dito.

> **Convenção:** "célula 7" significa a **sétima célula de código**. O Colab
> intercala células de texto, que não contam. Cada seção numerada do notebook
> corresponde a uma célula de código.

---

## Visão geral

| Etapa | Onde | Tempo |
| --- | --- | --- |
| 0. Preparar a máquina | local | ~10 min |
| 1. Treino (notebook 01) | Colab | 30 min a 6 h, conforme a configuração |
| 2. Exportação (notebook 02) | Colab | ~25 min |
| 3. Registro e avaliação | local | ~20 min |
| 4. Vídeo | local | ~1 h de gravação |

As etapas 1 e 2 exigem GPU com CUDA e por isso rodam no Colab. Todo o resto roda
no seu computador.

---

## Etapa 0 — Preparar a máquina local

### 0.1 Clonar e instalar

```bash
git clone https://github.com/alexandreccarmo/fia_tech3.git
cd fia_tech3
make setup
```

O `make setup` cria o `.venv` com Python 3.12, instala as dependências, gera o
`.env` a partir do `.env.example` e roda a suíte de testes.

### 0.2 Conferir o ambiente

```bash
make ambiente
```

Apresenta uma tabela item a item. **AVISO** em artefatos de etapas seguintes é
esperado; **FALHA** precisa ser resolvido antes de continuar.

### 0.3 Preparar os dados

```bash
make dados
make indexar
```

O primeiro baixa o PubMedQA, anonimiza, cura e gera o corpus hospitalar
sintético. O segundo constrói o índice FAISS. Ambos já vêm versionados no
repositório — rode apenas se quiser reproduzir do zero.

### 0.4 Contas necessárias

| Conta | Para quê |
| --- | --- |
| Google | Acessar o Colab |
| Hugging Face | Baixar o modelo base e publicar o GGUF |

No Hugging Face, crie um token de **escrita** em
*Settings → Access Tokens → Create new token → Write*. Ele será usado no segundo
notebook.

### 0.5 Aceitar a licença do modelo

O Llama 3.2 é *gated*: exige aceite. Acesse
[huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
preencha o formulário e envie.

**A aprovação não é instantânea.** Pode levar de minutos a horas, e não há como
consultar a fila. O aceite fica vinculado à **conta do Hugging Face**, não à
conta Google — trocar de conta Google não o invalida.

Se não quiser esperar, o notebook traz `Qwen/Qwen2.5-3B-Instruct` como
alternativa aberta, comentada na célula 7. Nesse caso **use o mesmo modelo no
notebook 02**.

---

## Como se chega ao notebook no Colab

Os links abaixo abrem os notebooks direto do GitHub. Vale entender como eles são
formados, porque quem trabalhar sobre uma cópia própria vai precisar montar os
seus.

**Não há geração nem integração.** O Colab tem um carregador de GitHub que
funciona por padrão de endereço: é a URL do arquivo no GitHub com `github.com/`
trocado por `colab.research.google.com/github/`.

```
https://github.com/USUARIO/REPO/blob/BRANCH/CAMINHO.ipynb

https://colab.research.google.com/github/USUARIO/REPO/blob/BRANCH/CAMINHO.ipynb
```

O Colab baixa aquele `.ipynb` — que é apenas JSON versionado como qualquer outro
arquivo — e o abre no editor. Funciona sem login porque o repositório é público.

O notebook, sozinho, não traz o código do projeto. Quem traz é a célula 3, que
executa `git clone` do repositório dentro da máquina virtual e acrescenta `src/`
ao caminho de importação do Python. É por isso que a célula seguinte consegue
fazer `from medgraph.finetune import colab_utils`. O dataset de treino vem no
mesmo clone, porque também está versionado.

O caminho é de mão única: o Colab lê do GitHub e nunca escreve de volta sozinho.
O resultado do treino retorna como download manual do `.zip`.

```
  sua maquina              GitHub                VM do Google (Colab)
  -----------              ------                --------------------
  edita codigo
  git commit
  git push     ------->    repositorio
                           publico
                              |
                              |  (1) Colab le o .ipynb pela URL
                              +--------------------------->  notebook aberto
                              |                                   |
                              |  (2) celula 3: git clone          |
                              +--------------------------->  codigo + dataset
                                                                  |
                                                             (3) treina na GPU
                                                                  |
  medgraph-adapter.zip  <-----------------------------------  (4) download
```

### Para uma cópia própria do projeto

Quem for modificar o projeto precisa que a célula 3 clone **o repositório dele** —
senão o Colab treinaria com o código de outra pessoa.

**1.** Publique a cópia no GitHub como repositório **público**.

**2.** Na célula 3, aponte a variável `REPO` para ela:

```python
REPO = "https://github.com/NOVO_USUARIO/NOVO_REPO.git"
```

E ajuste o caminho do clone, se o nome do diretório mudar.

**3.** Monte o link trocando as partes do endereço:

```
https://colab.research.google.com/github/NOVO_USUARIO/NOVO_REPO/blob/main/notebooks/colab/01_finetune_qlora_pubmedqa.ipynb
```

**4.** Repita para o `02_exportar_gguf.ipynb`.

### Sem montar a URL à mão

Dentro do Colab: **Arquivo → Abrir notebook → aba GitHub**, digite
`usuario/repositorio` e escolha o notebook na lista.

### Se o repositório for privado

A mesma aba GitHub tem a opção **Incluir repositórios privados**, que pede
autorização da conta. Mas o link direto deixa de funcionar para quem não tiver
acesso, e a célula do `git clone` também falharia — ela roda sem credenciais
dentro da VM. É por isso que este projeto é público: é o que torna "treinar"
equivalente a "abrir um link".

---

## Etapa 1 — Notebook 01: o treino

Abra:

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/colab/01_finetune_qlora_pubmedqa.ipynb
```

Ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução →
T4 GPU → Salvar**.

### Célula 1 — Verifica a GPU

Instantâneo. Deve imprimir `Tesla T4`.

Se disser que não há GPU, são duas causas: o runtime está sem acelerador (resolva
no menu acima), ou a cota diária acabou. Para distinguir, rode `!nvidia-smi`: se
o comando não existir e o menu já mostrar T4, é cota.

### Célula 2 — Instala as bibliotecas

~3 minutos.

**Depois dela, reinicie a sessão:** *Ambiente de execução → Reiniciar sessão*. E
continue **da célula 3**, não do começo. Este é o ponto onde mais gente tropeça.

### Célula 3 — Clona o repositório

~30 segundos. Traz o código e o dataset de treino para dentro da VM.

Ela só clona **se a pasta ainda não existir**. Para forçar código atualizado:
`!rm -rf /content/fia_tech3` antes.

### Célula 4 — Confere GPU e versões

Instantâneo. O aviso "GPU sem suporte a bfloat16 (esperado na T4)" é normal.

### Célula 5 — Login no Hugging Face

~1 minuto. Ela imprime um **código de dispositivo**. Abra
[hf.co/oauth/device](https://hf.co/oauth/device) em outra aba, já logado na conta
que tem o aceite, digite o código **com o hífen** e autorize.

O código expira em poucos minutos; se perder o prazo, reexecute a célula.

### Célula 6 — Carrega o dataset

~20 segundos. Mostra um exemplo completo, para conferência.

### Ajuste opcional — reduzir o treino

Se a configuração padrão não couber na sua cota (ver *Quanto tempo leva*, abaixo),
insira **uma célula nova entre a 6 e a 7** e cole:

```python
EXEMPLOS = 2000

colab_utils.CONFIG_PADRAO["num_train_epochs"]  = 1
colab_utils.CONFIG_PADRAO["logging_steps"]     = 5
colab_utils.CONFIG_PADRAO["eval_steps"]        = 25
colab_utils.CONFIG_PADRAO["save_steps"]        = 25
colab_utils.CONFIG_PADRAO["save_total_limit"]  = 2

dados["train"]      = dados["train"].shuffle(seed=42).select(range(min(EXEMPLOS, len(dados["train"]))))
dados["validation"] = dados["validation"].select(range(min(64, len(dados["validation"]))))

lote = (colab_utils.CONFIG_PADRAO["per_device_train_batch_size"]
        * colab_utils.CONFIG_PADRAO["gradient_accumulation_steps"])
print(f"{len(dados['train'])} exemplos | ~{len(dados['train'])//lote} passos "
      f"| ~{len(dados['train'])//lote*45/60:.0f} min")
```

Duas regras ao mexer nesses valores:

- `save_steps` precisa ser **múltiplo** de `eval_steps`, senão o
  `load_best_model_at_end` recusa a configuração;
- reduza `eval_steps` junto com o treino. Com o padrão de 100, um treino de 26
  passos não avaliaria nenhuma vez, e a curva da célula 11 sairia sem a linha de
  validação.

O `shuffle(seed=42)` é determinístico: a mesma amostra sai em toda execução, o
que torna o treino reproduzível e permite retomar de checkpoint.

### Célula 7 — Baixa e quantiza o modelo base

3 a 8 minutos. Baixa ~6 GB e quantiza para 4 bits em memória.

É aqui que se escolhe o modelo. A linha ativa deve ser a que você quer:

```python
MODELO_BASE = "meta-llama/Llama-3.2-3B-Instruct"   # gated — exige aceite
# MODELO_BASE = "Qwen/Qwen2.5-3B-Instruct"         # aberto
```

Ao terminar, imprime:

```
Parâmetros: 3.21 B
VRAM ocupada: 2.09 GB
```

**Confira a contagem de parâmetros:** 3,21 B é o Llama-3.2-3B; 3,09 B é o
Qwen2.5-3B. Se não for o modelo que você escolheu, pare — o resto da execução
seria sobre a arquitetura errada. A VRAM bem abaixo dos 6 GB baixados confirma
que a quantização funcionou.

Falha com `GatedRepoError` ou `403` significa que a conta não tem o aceite.

### Célula 8 — Configura o LoRA

Instantâneo. Prepara o modelo e informa quantos parâmetros treinarão.

Ela **não** chama `get_peft_model` de propósito: quem aplica o LoRA é o
`SFTTrainer`, na célula seguinte. Aplicar duas vezes empilha adaptadores e
derruba o treino com um erro que não menciona PEFT.

### Célula 9 — Monta o treinador

Instantâneo, mas é a célula de decisões.

**Ligue o Drive** se o treino for durar mais de meia hora:

```python
USAR_DRIVE = True
```

Custa ~1 minuto de autorização e ~1,5 GB do seu Drive, e é o que faz os
checkpoints sobreviverem à reciclagem da VM, a quedas de internet e a quedas de
energia. Numa rodada longa, vale sempre.

Ao final ela imprime **quantos passos** serão dados. Confira que bate com o que
você configurou — se você reduziu o dataset e ela imprime o número cheio, o
ajuste não foi aplicado.

A mensagem `Argumentos nao suportados ... IGNORADOS: [...]` é informativa: a
camada de compatibilidade removeu o que a versão instalada do `trl` recusa.

### Célula 10 — O treino

**~45 segundos por passo** numa T4. Multiplique pelo número de passos que a
célula 9 imprimiu.

A barra mostra `[passo/total tempo_decorrido < tempo_restante]`. O total cobre
**todas** as épocas, não uma.

A tabela de perdas aparece a cada `eval_steps`. O que se espera: `Training Loss` e
`Validation Loss` caindo juntas. Validação subindo enquanto treino cai é
sobreajuste.

**Mantenha a aba aberta.** E leia *Quando algo dá errado*, abaixo, antes de reagir
a qualquer desconexão.

### Célula 11 — Curva de perda

Instantâneo. Desenha treino e validação no mesmo gráfico e informa a redução
percentual da perda de validação.

### Célula 12 — Testa a adesão ao formato

~1 minuto. Gera resposta para 5 exemplos do conjunto de teste e verifica duas
coisas em cada uma: se começa com `Decisão: yes|no|maybe` e se cita `[E1]`.

**Espera-se 4 ou 5 de 5.** Menos que isso significa treino insuficiente — mas só
julgue por aqui se o treino foi completo. Num treino curto de dezenas de passos,
formato baixo é esperado e não indica erro.

### Célula 13 — Grava o adapter

~10 segundos. Salva em `models/adapters/medgraph-llama32-3b-lora`: os pesos do
adapter, o tokenizador, a curva de perda e o `cartao_de_treino.json` — que
registra modelo base, hiperparâmetros, versões, GPU, duração e perdas.

Sem esse cartão o adapter seria um binário sem procedência.

### Célula 14 — Baixa o `.zip`

~20 segundos. **Guarde este arquivo.** Ele é o produto do treino.

---

## Etapa 2 — Notebook 02: exportação

Abra:

```
https://colab.research.google.com/github/alexandreccarmo/fia_tech3/blob/main/notebooks/colab/02_exportar_gguf.ipynb
```

Se rodar na **mesma sessão** do notebook 01, o adapter já está em disco. Em sessão
nova, a célula 3 pede o upload do `.zip`.

### Célula 1 — Ambiente

Informa se há GPU. **Nada aqui exige CUDA**, mas use a **T4** mesmo assim.

A fusão da célula 5 carrega o modelo base em float16 — ~6,4 GB — e cria cópias
durante o merge, contra os ~12,7 GB de RAM do Colab gratuito. A folga é estreita,
e estourar a memória **derruba a sessão inteira**, levando junto o upload do
adapter e o login. Com a GPU, o modelo vai para a VRAM e a RAM fica livre.

A conversão do `llama.cpp`, mais adiante, é CPU de qualquer forma.

### Célula 2 — Dependências e repositório

~3 minutos. Instala, remove o `torchao` e clona.

A remoção do `torchao` não é capricho: o Colab traz a versão 0.10 pré-instalada e
o `peft` recente exige 0.16 ou superior. A incompatibilidade não aparece aqui —
ela estoura na célula 5, dentro de `PeftModel.from_pretrained`, com um
`ImportError` que fala de `torchao` e não menciona o adapter. Este notebook não
usa a biblioteca: a quantização é do `llama.cpp` e a fusão é em float16.

### Célula 3 — O adapter

Localiza o adapter. Se não encontrar, abre o seletor de upload para o
`medgraph-adapter.zip`.

Ela imprime o **cartão de treino**: modelo base, exemplos, duração e perdas.
Confira o `modelo base` — é a última chance de perceber que se está fundindo na
arquitetura errada.

### Célula 4 — Autenticação

Mesmo fluxo de código de dispositivo do notebook 01. Aqui o token precisa ser de
**escrita**, porque o notebook publica no Hub.

### Célula 5 — Fusão

**Ajuste o `MODELO_BASE` para o mesmo do notebook 01.** É o erro mais caro
possível neste ponto: fundir num modelo diferente não falha — gera um GGUF que
carrega, responde, e responde mal.

A fusão soma as matrizes do adapter aos pesos do modelo base e produz um modelo
único e completo. Roda em CPU e consome bastante RAM.

### Célula 6 — llama.cpp

Clona e compila o conversor e o quantizador. Alguns minutos.

### Célula 7 — Conversão e quantização

Converte o modelo fundido para GGUF em f16 e depois quantiza para **Q4_K_M** —
o nível que equilibra tamanho e qualidade.

### Célula 8 — Publicação

Cria o repositório no Hugging Face e envia o GGUF junto com o `Modelfile`. Ao
final imprime o nome exato do repositório e a linha para o seu `.env`:

```
REPO_GGUF_HF=seu-usuario/medgraph-llama32-3b-gguf
```

**Anote esse valor.**

### Célula 9 — Alternativa

Download direto do GGUF, caso você prefira não publicar no Hub.

---

## Etapa 3 — De volta à máquina

### 3.1 Descompactar o adapter

```bash
cd ~/Desktop/FIAP/projeto/tech_challenge_fiap3/fia_tech3
unzip ~/Downloads/medgraph-adapter.zip -d models/adapters/
```

Vêm junto o `cartao_de_treino.json` e a `curva_de_perda.png` — os dois artefatos
que o vídeo mostra no bloco de fine-tuning. Para deixar o gráfico junto dos
demais:

```bash
cp models/adapters/medgraph-llama32-3b-lora/curva_de_perda.png docs/graficos/
```

### 3.2 Ajustar o `.env`

```
REPO_GGUF_HF=seu-usuario/medgraph-llama32-3b-gguf
HF_TOKEN=hf_...
OLLAMA_MODEL=medgraph
```

Só troque `OLLAMA_MODEL` para `medgraph` **depois** de registrar o modelo no
passo seguinte. Antes disso, o valor correto continua sendo `medgraph-base`.

### 3.3 Registrar no Ollama

```bash
make modelo -- --ajustado
```

Baixa o GGUF do Hub e registra no Ollama. O separador `--` é obrigatório: é ele
que faz a flag chegar ao script.

Confira:

```bash
ollama list
ollama run medgraph "Qual a conduta inicial na suspeita de sepse?"
```

### 3.4 Reavaliar

```bash
make avaliar
make relatorio
```

A tabela comparativa passa a ter a coluna do modelo ajustado ao lado do modelo
base e do `gpt-4o-mini`. O relatório técnico é regenerado com os números novos e
com os dados do cartão de treino.

Para uma passada rápida de conferência: `make avaliar -- --rapido` (30 casos).

### 3.5 Conferir tudo

```bash
make testes
make lint
make rastreabilidade
```

---

## Etapa 4 — O vídeo

```bash
make app
```

Abre o painel Streamlit, que é o cenário da maior parte da gravação. O roteiro
cronometrado, com a divisão entre os integrantes, está em
[`roteiro_video.md`](roteiro_video.md).

Dos 8 blocos do roteiro, **6 não dependem do modelo ajustado** — inclusive o
Bloco 5, a demonstração ao vivo, que é o núcleo. Eles podem ser gravados a
qualquer momento, com o `medgraph-base`. Só os blocos 3 (fine-tuning) e 7
(avaliação) precisam do treino concluído.

---

## Quanto tempo leva o treino

Medido: **~45 segundos por passo** numa T4 gratuita.

O número de passos é `exemplos × épocas ÷ 16`. Com o dataset completo de 3.871
exemplos e 2 épocas, são 484 passos — **quase 6 horas**, mais do que a cota
diária do plano gratuito, que fica entre 3 e 4 horas.

| Configuração | Passos | Tempo |
| --- | ---: | ---: |
| Padrão (2 épocas, 3.871 ex.) | 484 | ~6 h |
| 1 época | 242 | ~3 h |
| 1 época, 2.000 exemplos | 125 | ~1 h 35 |
| 1 época, 800 exemplos | 50 | ~40 min |

Reduzir o dataset é preferível a reduzir o `max_seq_length`: os exemplos têm ~900
tokens em média, e cortar para 512 truncaria a resposta esperada em metade deles.

---

## Quando algo dá errado

### A conexão caiu

**Não reinicie a sessão por reflexo.** Perder o navegador não é perder o treino: a
VM continua executando na nuvem, indiferente ao seu computador.

1. Recarregue a página. Isso reconecta a interface ao mesmo kernel.
2. Se o "Conectando" não sair, feche **todas** as abas do Colab e reabra em janela
   anônima — extensões de bloqueio derrubam o websocket com frequência.
3. Confirme em *Ambiente de execução → Gerenciar sessões* se a sessão continua
   ativa. "Última execução: há 0 minuto" significa que ela está trabalhando agora.

**Nunca abra duas abas do mesmo notebook.** Elas disputam a sessão e é uma causa
comum de "Conectando" eterno.

### Acompanhar o treino sem o Colab

Se você ligou o `USAR_DRIVE`, abra [drive.google.com](https://drive.google.com) e
navegue até `Meu Drive / medgraph / saida_treino`. As pastas `checkpoint-N` e
suas horas de modificação mostram o progresso **sem depender de o Colab
conectar**.

Só aparecem os últimos checkpoints porque `save_total_limit` limita quantos ficam
em disco. Não é perda de dado.

### Retomar de um checkpoint

Rode as células 3 a 9 — **incluindo o bloco de ajuste**, se você usou um —, com o
mesmo valor de `USAR_DRIVE`. Confira que a célula 9 imprime o mesmo número de
passos de antes. Então, na célula 10:

```python
resultado = treinador.train(resume_from_checkpoint=True)
```

### Recuperar o adapter sem rodar a célula 13

Cada checkpoint contém os pesos treinados. Se a sessão morreu depois do treino
mas antes de salvar, o resultado **não se perdeu** — as células 11 a 14 apenas
formatam e empacotam o que o checkpoint já tem.

No Drive, clique com o botão direito na pasta do checkpoint mais avançado e
escolha **Fazer download**. Depois, na sua máquina:

```bash
make recuperar-adapter -- --checkpoint ~/Downloads/checkpoint-125
```

O comando copia o adapter e o tokenizador para
`models/adapters/medgraph-llama32-3b-lora`, redesenha a curva de perda a partir
do `trainer_state.json` e grava o cartão de treino. Aceita tanto a pasta de um
checkpoint quanto a que contém vários — nesse caso escolhe o de maior número de
passos.

Passe também `--modelo-base`, que o notebook de exportação lê:

```bash
make recuperar-adapter -- --checkpoint ~/Downloads/checkpoint-125 \
    --modelo-base meta-llama/Llama-3.2-3B-Instruct
```

O cartão gerado por essa via registra explicitamente que GPU, versões de
biblioteca e duração **não foram capturadas** — esses dados só existiam na
sessão do Colab. Um cartão de procedência que adivinha é pior do que um cartão
incompleto.

O que se perde nesse caminho é o teste de formato da célula 12. Ele pode ser
refeito na máquina depois de registrar o modelo no Ollama.

### A cota de GPU acabou

O Colab não recusa a conexão: conecta em CPU e mantém "T4 GPU" no menu, o que faz
parecer problema de configuração. Confirme com `!nvidia-smi`.

Não há o que consertar. A cota volta em algumas horas. Alternativas: outra conta
Google (a cota é por conta, e o aceite do Hugging Face continua valendo), ou o
Colab Pro.

### Erros mais comuns

| Sintoma | Causa | O que fazer |
| --- | --- | --- |
| `Nenhuma GPU disponível` | Sem acelerador, ou cota | `!nvidia-smi` distingue |
| `GatedRepoError` / `403` na célula 7 | Conta sem aceite do Llama | Aceite, ou use o Qwen |
| `ModuleNotFoundError` na célula 4 | Sessão não reiniciada após a 2 | Reinicie e siga da 3 |
| `CUDA out of memory` | Sequência longa demais | Antes da 9: `colab_utils.CONFIG_PADRAO["max_seq_length"] = 768` |
| `save_steps` não é múltiplo de `eval_steps` | Ajuste manual incoerente | Iguale os dois valores |
| Formato baixo na célula 12 | Treino insuficiente | Mais épocas ou mais exemplos |
| `FileNotFoundError: nvidia-smi` na célula 1 do notebook 02 | Runtime sem GPU, que aqui é o esperado | Corrigido; em notebook antigo, pule a célula |
| `ImportError: incompatible version of torchao` na célula 5 do notebook 02 | `torchao` 0.10 do Colab contra o mínimo do `peft` | `%pip uninstall -y -q torchao` e reexecute |
| Sessão morre na célula 5 do notebook 02 | RAM esgotada durante a fusão | Troque para T4 GPU e use `device_map="auto"` |

---

## Checklist de conclusão

- [ ] `models/adapters/medgraph-llama32-3b-lora/` com adapter e cartão de treino
- [ ] GGUF publicado no Hugging Face Hub
- [ ] `.env` com `REPO_GGUF_HF` e `OLLAMA_MODEL=medgraph`
- [ ] `ollama list` mostrando `medgraph`
- [ ] `make avaliar` com a coluna do modelo ajustado
- [ ] `make relatorio` regenerado
- [ ] `make testes` e `make lint` limpos
- [ ] Vídeo de até 15 minutos gravado (`REQ-E4`)
- [ ] Repositório com tudo commitado e enviado
