"""
[REQ-1] Montagem do dataset de fine-tuning supervisionado.

O QUE FAZ:
    Converte os quatro corpora curados em um unico conjunto de exemplos no
    formato de conversa (system / user / assistant), pronto para ser enviado
    ao notebook do Colab.

AS QUATRO FONTES E O QUE CADA UMA ENSINA:

    1. PubMedQA artificial (~2.500)
       Volume. Ensina o FORMATO da tarefa - ler evidencia, decidir, citar -
       antes de o modelo ver os exemplos bons. Rotulos automaticos, so
       yes/no.

    2. PubMedQA anotado por especialistas (~449)
       Qualidade. Traz a classe "maybe", que nao existe no artificial, e
       decisoes revisadas por humanos. Poucos exemplos, alto valor.

    3. FAQ do corpo medico (~200, PT-BR)
       Dominio e idioma. O PubMedQA e todo em ingles; sem esta fonte, o
       modelo responderia em ingles a perguntas em portugues. Tambem ensina
       a citar protocolo interno com [P#].

    4. Modelos de documentos (~10 x variacoes)
       Formato institucional e, principalmente, o limite de seguranca: toda
       receita ou prescricao sai marcada como rascunho pendente de assinatura.

POR QUE MISTURAR TAREFAS DIFERENTES:
    Um modelo treinado so em decisao yes/no/maybe vira um classificador e
    perde a capacidade de conversar. Um treinado so em FAQ nao aprende a se
    ancorar em evidencia. A mistura mantem as duas competencias, e a
    proporcao entre elas e uma decisao de projeto registrada no relatorio.

O CONJUNTO DE TESTE NAO ENTRA AQUI.
    Os 500 exemplos separados na curadoria ficam intocados ate a Etapa 4.
    Ha uma verificacao explicita no fim deste modulo garantindo que nenhum
    pubid do teste vazou para o treino.

Uso:
    python -m medgraph.finetune.preparar_dataset_sft
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.chains import prompts
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

# Proporcao da validacao dentro do conjunto de treino final.
PROPORCAO_VALIDACAO_SFT = 0.05

# -----------------------------------------------------------------------------
# REPETICAO DO CONJUNTO ANOTADO POR ESPECIALISTAS, POR CLASSE
# -----------------------------------------------------------------------------
# O PROBLEMA:
#   O conjunto artificial do PubMedQA nao tem NENHUM exemplo "maybe" - os
#   rotulos automaticos so produzem yes/no. Todo o "maybe" do treino vem dos
#   449 exemplos anotados por especialistas, e la ele representa apenas 11%.
#   Num dataset final de milhares de exemplos, "maybe" ficaria abaixo de 1%.
#
#   A consequencia e concreta e mensuravel: o modelo simplesmente nunca
#   preveria "maybe". Como a avaliacao usa macro-F1 - que da o mesmo peso as
#   tres classes - uma classe com F1 igual a zero limita a metrica a cerca de
#   0,67 por construcao, independentemente de quao bem as outras duas vao.
#
# A CORRECAO:
#   Repetimos os exemplos de especialista com fator INVERSO a frequencia da
#   classe. Escolhemos repetir a classe rara em vez de descartar a comum
#   porque o conjunto de especialista ja e pequeno (449) e joga-lo fora seria
#   desperdicar o dado de maior qualidade que temos.
#
# O RISCO, ASSUMIDO E MONITORADO:
#   Repetir 49 exemplos oito vezes convida a memorizacao. E um risco aceitavel
#   porque a avaliacao da Etapa 4 roda sobre 500 exemplos jamais vistos: se
#   houver memorizacao em vez de aprendizado, a metrica de teste vai denunciar.
REPETICOES_POR_CLASSE: dict[str, int] = {"yes": 2, "no": 3, "maybe": 8}

# Tamanho maximo, em caracteres, do trecho de protocolo injetado como
# contexto num exemplo de FAQ. Cerca de 1.400 caracteres cabem
# confortavelmente na janela de 1.024 tokens usada no treino, junto com a
# pergunta e a resposta.
MAX_CARACTERES_TRECHO_PROTOCOLO = 1400


def _mensagens(usuario: str, assistente: str) -> dict[str, Any]:
    """Um exemplo no formato de conversa esperado pelo SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": prompts.SISTEMA},
            {"role": "user", "content": usuario},
            {"role": "assistant", "content": assistente},
        ]
    }


# =============================================================================
# FONTE 1 e 2 - PubMedQA
# =============================================================================
def _exemplos_pubmedqa(
    registros: list[dict[str, Any]],
    origem: str,
    *,
    repeticoes_por_classe: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Converte registros do PubMedQA em exemplos de conversa.

    Args:
        repeticoes_por_classe: quantas vezes repetir cada exemplo conforme o
            rotulo. Usado apenas no conjunto de especialista, para compensar
            a ausencia de "maybe" no conjunto artificial. Ver
            REPETICOES_POR_CLASSE.
    """
    exemplos: list[dict[str, Any]] = []
    for r in registros:
        usuario = prompts.usuario_decisao(r["pergunta"], r["contexto"])
        assistente = prompts.assistente_decisao(r["decisao"], r["resposta_longa"])

        vezes = (repeticoes_por_classe or {}).get(r["decisao"], 1)
        for _ in range(vezes):
            exemplo = _mensagens(usuario, assistente)
            exemplo["_origem"] = origem
            exemplo["_rotulo"] = r["decisao"]
            exemplo["_pubid"] = r.get("pubid")
            exemplos.append(exemplo)
    return exemplos


# =============================================================================
# FONTE 3 - FAQ do corpo medico
# =============================================================================
def _carregar_protocolos(cfg: Settings) -> dict[str, dict[str, str]]:
    """
    Le os protocolos internos e indexa por identificador.

    Guardamos o texto integral e tambem as secoes separadas, para conseguir
    escolher o trecho mais relevante para cada pergunta.
    """
    protocolos: dict[str, dict[str, str]] = {}
    pasta = cfg.dir_dados_sinteticos / "protocolos"
    if not pasta.is_dir():
        return protocolos

    for arquivo in sorted(pasta.glob("PROT-*.md")):
        texto = arquivo.read_text(encoding="utf-8")
        identificador = arquivo.name.split("-")[0] + "-" + arquivo.name.split("-")[1]

        titulo_match = re.search(r"^titulo:\s*(.+)$", texto, re.MULTILINE)
        titulo = titulo_match.group(1).strip() if titulo_match else arquivo.stem

        protocolos[identificador] = {"titulo": titulo, "texto": texto, "arquivo": arquivo.name}
    return protocolos


def _trecho_relevante(texto_protocolo: str, pergunta: str) -> str:
    """
    Escolhe a parte do protocolo que melhor responde a pergunta.

    POR QUE FAZER ISSO NA CONSTRUCAO DO DATASET:
        Em producao, o contexto do modelo vem do RAG - um trecho recuperado,
        nao o protocolo inteiro. Se treinassemos com o documento completo, o
        modelo aprenderia a esperar um contexto longo e organizado, e teria
        desempenho pior quando recebesse os quatro trechos curtos que o
        recuperador realmente devolve.

        Reproduzir aqui a mesma forma de contexto que existira na inferencia
        e o que faz o treino transferir para o uso real.

    O criterio e sobreposicao de palavras entre a pergunta e cada secao -
    uma recuperacao simples, suficiente porque ja sabemos qual protocolo e
    o correto (vem do campo protocolo_ref da FAQ).
    """
    secoes = re.split(r"\n## ", texto_protocolo)
    palavras_pergunta = {
        p for p in re.findall(r"\w{4,}", pergunta.lower())
    }

    melhor_secao, melhor_pontuacao = "", -1
    for secao in secoes:
        if secao.lstrip().startswith(("Aviso", "9. Referências", "9. Referencias")):
            continue
        palavras_secao = set(re.findall(r"\w{4,}", secao.lower()))
        pontuacao = len(palavras_pergunta & palavras_secao)
        if pontuacao > melhor_pontuacao:
            melhor_secao, melhor_pontuacao = secao, pontuacao

    trecho = ("## " + melhor_secao).strip()
    if len(trecho) > MAX_CARACTERES_TRECHO_PROTOCOLO:
        trecho = trecho[:MAX_CARACTERES_TRECHO_PROTOCOLO].rsplit("\n", 1)[0] + "\n[...]"
    return trecho


def _exemplos_faq(cfg: Settings) -> list[dict[str, Any]]:
    caminho = cfg.dir_dados_sinteticos / "faq_medicos.jsonl"
    if not caminho.exists():
        log.warning("faq_medicos.jsonl nao encontrado - pulando a fonte 3.")
        return []

    protocolos = _carregar_protocolos(cfg)
    exemplos: list[dict[str, Any]] = []

    with caminho.open(encoding="utf-8") as arquivo:
        linhas_faq = [linha.strip() for linha in arquivo if linha.strip()]

    for linha in linhas_faq:
        faq = json.loads(linha)
        ref = faq.get("protocolo_ref", "")
        protocolo = protocolos.get(ref)

        if protocolo:
            trecho = _trecho_relevante(protocolo["texto"], faq["pergunta"])
            contexto = prompts.montar_contexto(
                [{"marcador": "P1", "titulo": f"{ref} — {protocolo['titulo']}", "texto": trecho}]
            )
        else:
            contexto = prompts.montar_contexto(
                [{"marcador": "P1", "titulo": ref or "Protocolo interno", "texto": faq["resposta"]}]
            )

        resposta = faq["resposta"].strip()
        if faq.get("exige_validacao_humana"):
            resposta += (
                "\n\nEsta orientação envolve conduta terapêutica e depende de validação "
                "do médico responsável antes de qualquer prescrição."
            )
        resposta += "\nFontes: [P1]"

        exemplo = _mensagens(prompts.usuario_protocolo(faq["pergunta"], contexto), resposta)
        exemplo["_origem"] = "faq_medicos"
        exemplo["_rotulo"] = "faq"
        exemplos.append(exemplo)

    return exemplos


# =============================================================================
# FONTE 4 - Modelos de documentos
# =============================================================================
def _exemplos_documentos(cfg: Settings) -> list[dict[str, Any]]:
    pasta = cfg.dir_dados_sinteticos / "modelos_documentos"
    if not pasta.is_dir():
        log.warning("modelos_documentos/ nao encontrado - pulando a fonte 4.")
        return []

    exemplos: list[dict[str, Any]] = []
    for arquivo in sorted(pasta.glob("DOC-*.md")):
        texto = arquivo.read_text(encoding="utf-8")
        titulo = (re.search(r"^titulo:\s*(.+)$", texto, re.MULTILINE) or [None, arquivo.stem])[1]
        titulo = str(titulo).strip()
        identificador = arquivo.name.split("-")[0] + "-" + arquivo.name.split("-")[1]

        # Extrai a secao "3. Modelo (gabarito)", que e o que o assistente
        # deve saber reproduzir.
        gabarito = ""
        m = re.search(r"## 3\. Modelo.*?\n(.*?)(?=\n## 4\.)", texto, re.DOTALL)
        if m:
            gabarito = m.group(1).strip()

        if not gabarito:
            continue

        e_prescricao = identificador in {"DOC-002", "DOC-010"}
        resposta = gabarito
        if e_prescricao:
            resposta = (
                "RASCUNHO — sem validade até assinatura do médico responsável.\n\n"
                + gabarito
                + "\n\nEste rascunho não constitui prescrição. Nenhum sistema automatizado "
                  "pode emitir prescrição sem validação humana registrada."
            )
        resposta += "\nFontes: [P1]"

        contexto = prompts.montar_contexto(
            [{"marcador": "P1", "titulo": f"{identificador} — {titulo}", "texto": gabarito}]
        )

        # Tres formas de pedir o mesmo documento: o modelo precisa reconhecer
        # a intencao, nao decorar uma frase.
        for pedido in (
            f"Preciso do modelo de {titulo.lower()} do hospital.",
            f"Como devo estruturar um {titulo.lower()}?",
            f"Me mostre o gabarito institucional de {titulo.lower()}, por favor.",
        ):
            exemplo = _mensagens(prompts.usuario_documento(pedido, contexto), resposta)
            exemplo["_origem"] = "modelos_documentos"
            exemplo["_rotulo"] = "documento"
            exemplos.append(exemplo)

    return exemplos


# =============================================================================
# MONTAGEM
# =============================================================================
def _ler_jsonl(caminho: Path) -> list[dict[str, Any]]:
    """
    Le um arquivo JSON Lines.

    ATENCAO ao detalhe da implementacao: iteramos o objeto de arquivo em vez
    de usar `read_text().splitlines()`. O splitlines() quebra em qualquer
    separador de linha Unicode - \x0b, \x0c, \x85, \u2028, \u2029 - e todos
    eles sao caracteres validos dentro de uma string JSON, que o json.dumps
    nao escapa. Um unico abstract cientifico contendo um desses caracteres
    partiria a linha em duas e quebraria o parse do arquivo inteiro.
    Em modo texto, a iteracao do arquivo so reconhece \n, \r e \r\n.
    """
    if not caminho.exists():
        return []
    registros: list[dict[str, Any]] = []
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    return registros


def _gravar_jsonl(registros: list[dict[str, Any]], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for r in registros:
            # Os campos auxiliares com prefixo "_" servem so para as
            # estatisticas; nao vao para o arquivo de treino.
            arquivo.write(
                json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n"
            )


def executar(cfg: Settings | None = None) -> dict[str, Any]:
    """Monta sft_train.jsonl e sft_valid.jsonl em data/processed/."""
    cfg = cfg or obter_settings()
    proc = cfg.dir_dados_processados

    with etapa("preparar_dataset_sft"):
        exemplos: list[dict[str, Any]] = []
        exemplos += _exemplos_pubmedqa(_ler_jsonl(proc / "pubmedqa_artificial.jsonl"), "pubmedqa_artificial")
        exemplos += _exemplos_pubmedqa(
            _ler_jsonl(proc / "pubmedqa_treino.jsonl"),
            "pubmedqa_especialista",
            repeticoes_por_classe=REPETICOES_POR_CLASSE,
        )
        exemplos += _exemplos_faq(cfg)
        exemplos += _exemplos_documentos(cfg)

        # -----------------------------------------------------------------
        # VERIFICACAO DE VAZAMENTO DO CONJUNTO DE TESTE
        # -----------------------------------------------------------------
        # Se um unico pubid do teste aparecer no treino, toda a avaliacao da
        # Etapa 4 perde o sentido. Falhamos alto e cedo em vez de descobrir
        # isso depois de treinar.
        pubids_teste = {
            r.get("pubid") for r in _ler_jsonl(proc / "pubmedqa_teste.jsonl") if r.get("pubid")
        }
        pubids_treino = {e.get("_pubid") for e in exemplos if e.get("_pubid")}
        vazamento = pubids_teste & pubids_treino
        if vazamento:
            raise RuntimeError(
                f"VAZAMENTO: {len(vazamento)} pubid(s) do conjunto de teste apareceram no "
                f"treino. A avaliacao seria invalida. Exemplos: {sorted(vazamento)[:5]}"
            )
        log.info(
            "Verificacao de vazamento: nenhum dos %d pubids de teste aparece no treino.",
            len(pubids_teste),
        )

        # -----------------------------------------------------------------
        # Embaralhar e dividir
        # -----------------------------------------------------------------
        rng = random.Random(cfg.semente_aleatoria)
        rng.shuffle(exemplos)

        corte = max(1, int(len(exemplos) * PROPORCAO_VALIDACAO_SFT))
        validacao, treino = exemplos[:corte], exemplos[corte:]

        _gravar_jsonl(treino, proc / "sft_train.jsonl")
        _gravar_jsonl(validacao, proc / "sft_valid.jsonl")

    # -------------------------------------------------------------------------
    # Estatisticas
    # -------------------------------------------------------------------------
    por_origem = Counter(e["_origem"] for e in exemplos)
    por_rotulo = Counter(e["_rotulo"] for e in exemplos)
    caracteres = [
        sum(len(m["content"]) for m in e["messages"]) for e in exemplos
    ]

    proporcoes = {
        rotulo: round(qtd / max(1, len(exemplos)), 4) for rotulo, qtd in por_rotulo.items()
    }

    resumo = {
        "total": len(exemplos),
        "proporcao_por_rotulo": proporcoes,
        "repeticoes_aplicadas": REPETICOES_POR_CLASSE,
        "treino": len(treino),
        "validacao": len(validacao),
        "por_origem": dict(por_origem),
        "por_rotulo": dict(por_rotulo),
        "caracteres_por_exemplo": {
            "media": round(sum(caracteres) / max(1, len(caracteres))),
            "minimo": min(caracteres, default=0),
            "maximo": max(caracteres, default=0),
        },
        # Estimativa grosseira: ~4 caracteres por token. Serve para dimensionar
        # o tempo de treino no Colab antes de gastar a GPU.
        "tokens_estimados_total": round(sum(caracteres) / 4),
    }

    (proc / "relatorio_sft.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info("Dataset SFT montado:")
    log.info("  total ........... %d exemplos", resumo["total"])
    log.info("  treino .......... %d", resumo["treino"])
    log.info("  validacao ....... %d", resumo["validacao"])
    log.info("  por origem ...... %s", resumo["por_origem"])
    log.info("  por rotulo ...... %s", resumo["por_rotulo"])
    log.info("  proporcao ....... %s", {k: f"{v:.1%}" for k, v in proporcoes.items()})
    log.info("  tamanho medio ... %d caracteres", resumo["caracteres_por_exemplo"]["media"])
    log.info("  tokens estimados  ~%d", resumo["tokens_estimados_total"])

    registrar(
        TipoEvento.FIM_ETAPA,
        "Dataset de fine-tuning montado",
        etapa="preparar_dataset_sft",
        conclusao=True,
        **{k: v for k, v in resumo.items() if isinstance(v, (int, str))},
    )
    return resumo


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Dataset de fine-tuning", subtitulo="formato de conversa para o Colab")
    executar()
