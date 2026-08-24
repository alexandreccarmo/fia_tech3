"""
[REQ-E3] Avaliação comparativa dos sistemas.

O QUE FAZ:
    Roda o mesmo conjunto de teste — 500 exemplos anotados por especialistas,
    jamais vistos no treino — através de quatro sistemas, e produz a tabela
    comparativa que sustenta a seção de resultados do relatório técnico.

OS QUATRO SISTEMAS E O PAPEL DE CADA UM

    1. classe majoritária
       Responde sempre "yes". É o PISO. Sem ele, um modelo com 58% de
       accuracy parece razoável — até se descobrir que responder sempre a
       mesma coisa dá 55%.

    2. modelo base (medgraph-base)
       Llama-3.2-3B-Instruct servido com o MESMO prompt de sistema, a MESMA
       temperatura e o MESMO template do modelo ajustado. Essa igualdade de
       condições é o que torna a comparação honesta: qualquer diferença de
       configuração atribuiria ao treino um ganho que veio de outro lugar.

    3. modelo ajustado (medgraph)
       O resultado do fine-tuning. A diferença entre 2 e 3 é o que o trabalho
       precisa demonstrar.

    4. gpt-4o-mini
       TETO DE REFERÊNCIA, não concorrente. Um modelo cerca de cem vezes
       maior, para dar escala ao resultado: saber que o modelo ajustado ficou
       a poucos pontos de um modelo dessa dimensão diz mais do que o número
       absoluto sozinho.

REFERÊNCIA EXTERNA:
    O artigo original do PubMedQA reporta 78% de acurácia para especialistas
    humanos no mesmo conjunto. É o horizonte da tarefa.

CONTROLE DE CUSTO:
    Cada chamada à OpenAI passa pela trava de orçamento. A avaliação do
    gpt-4o-mini roda sobre uma amostra menor por padrão — 500 exemplos
    custariam cerca de US$ 0,20, o que caberia no orçamento, mas não
    acrescentaria informação suficiente para justificar o gasto.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.avaliacao.metricas import (
    ResultadoAvaliacao,
    baseline_classe_majoritaria,
    extrair_decisao,
    tabela_comparativa,
)
from medgraph.chains import prompts
from medgraph.llm.custo import OrcamentoExcedidoError, contador
from medgraph.llm.provider import ProvedorIndisponivelError, obter_llm
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

# Amostra padrão para os modelos locais. 500 exemplos num modelo de 3B levam
# cerca de uma hora por sistema; 150 já dão intervalo de confiança suficiente
# para a comparação e mantêm a execução em torno de 15 minutos.
# Use --completo para rodar os 500.
AMOSTRA_PADRAO_LOCAL = 150

# Amostra para a OpenAI. Menor por causa do custo — ver a nota no cabeçalho.
AMOSTRA_PADRAO_OPENAI = 100


def carregar_teste(cfg: Settings | None = None, limite: int | None = None) -> list[dict[str, Any]]:
    """
    Carrega o conjunto de teste isolado na Etapa 1.

    A amostragem é sempre um PREFIXO da lista, e não um sorteio novo a cada
    execução: os registros já foram embaralhados uma vez, com semente fixa, na
    curadoria. Assim todos os sistemas veem exatamente os mesmos exemplos, na
    mesma ordem, em qualquer execução — condição necessária para que a
    comparação signifique alguma coisa.
    """
    cfg = cfg or obter_settings()
    caminho = cfg.dir_dados_processados / "pubmedqa_teste.jsonl"
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode primeiro: make dados")

    casos: list[dict[str, Any]] = []
    with caminho.open(encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                casos.append(json.loads(linha))
    return casos[:limite] if limite else casos


def avaliar_sistema(
    nome: str,
    llm: BaseChatModel,
    casos: Sequence[dict[str, Any]],
    *,
    mostrar_progresso: bool = True,
) -> ResultadoAvaliacao:
    """
    Executa um sistema sobre os casos de teste e coleta os resultados.

    Erros de execução (timeout, orçamento esgotado, servidor fora do ar) são
    contabilizados como falha daquele caso, sem interromper a avaliação —
    perder uma hora de execução por causa de um timeout isolado seria pior do
    que registrar o caso como não respondido.
    """
    resultado = ResultadoAvaliacao(sistema=nome)

    iterador: Any = casos
    if mostrar_progresso:
        try:
            from tqdm import tqdm

            iterador = tqdm(casos, desc=f"{nome:<28}", unit="caso", leave=True)
        except ImportError:
            pass

    for caso in iterador:
        mensagens = [
            SystemMessage(content=prompts.SISTEMA),
            HumanMessage(content=prompts.usuario_decisao(caso["pergunta"], caso["contexto"])),
        ]

        inicio = time.perf_counter()
        try:
            resposta = str(llm.invoke(mensagens).content)
        except OrcamentoExcedidoError:
            log.error("Orçamento esgotado durante a avaliação de %s. Interrompendo.", nome)
            break
        except Exception as exc:
            log.warning("Falha em um caso de %s: %s: %s", nome, type(exc).__name__, exc)
            resultado.erros += 1
            resposta = ""
        latencia = (time.perf_counter() - inicio) * 1000

        previsto, metodo = extrair_decisao(resposta)

        resultado.verdadeiros.append(caso["decisao"])
        resultado.previstos.append(previsto)
        resultado.metodos.append(metodo)
        resultado.latencias_ms.append(latencia)
        resultado.respostas.append(resposta)
        resultado.perguntas.append(caso["pergunta"])

    registrar(
        TipoEvento.FIM_ETAPA,
        f"Avaliação de {nome} concluída",
        etapa=f"avaliar:{nome}",
        conclusao=True,
        total=resultado.total,
        accuracy=round(resultado.accuracy, 4),
        macro_f1=round(resultado.macro_f1, 4),
        taxa_adesao_formato=round(resultado.taxa_adesao_formato, 4),
        erros=resultado.erros,
    )
    return resultado


def _modelos_no_ollama(cfg: Settings) -> list[str]:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=5) as r:
            return [m["name"].split(":")[0] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def executar(
    cfg: Settings | None = None,
    *,
    n_local: int = AMOSTRA_PADRAO_LOCAL,
    n_openai: int = AMOSTRA_PADRAO_OPENAI,
    incluir_openai: bool = True,
) -> dict[str, Any]:
    """
    Roda a avaliação completa e grava os resultados.

    Produz:
        docs/avaliacao_resultados.json   números completos, por sistema
        docs/graficos/*.png              gráficos do relatório
    """
    cfg = cfg or obter_settings()
    casos = carregar_teste(cfg, limite=n_local)
    disponiveis = _modelos_no_ollama(cfg)

    log.info(
        "Conjunto de teste: %d casos | distribuição: %s",
        len(casos),
        {r: sum(1 for c in casos if c["decisao"] == r) for r in ("yes", "no", "maybe")},
    )

    resultados: list[ResultadoAvaliacao] = []

    # -- 1. Piso -------------------------------------------------------------
    resultados.append(baseline_classe_majoritaria([c["decisao"] for c in casos]))

    # -- 2. Modelo base ------------------------------------------------------
    if "medgraph-base" in disponiveis:
        with etapa("avaliar:base"):
            cfg_base = cfg.model_copy(update={"ollama_model": "medgraph-base"})
            llm = obter_llm(cfg_base, origem="avaliacao.base", temperatura=0.0, provedor="ollama")
            resultados.append(avaliar_sistema("modelo base (Llama-3.2-3B)", llm, casos))
    else:
        log.warning("'medgraph-base' não registrado no Ollama — sistema 2 pulado.")

    # -- 3. Modelo ajustado --------------------------------------------------
    if cfg.ollama_model != "medgraph-base" and cfg.ollama_model in disponiveis:
        with etapa("avaliar:ajustado"):
            llm = obter_llm(cfg, origem="avaliacao.ajustado", temperatura=0.0, provedor="ollama")
            resultados.append(avaliar_sistema(f"modelo ajustado ({cfg.ollama_model})", llm, casos))
    else:
        log.warning(
            "Modelo ajustado ('%s') ainda não registrado. Execute os notebooks do "
            "Colab e depois 'make modelo --ajustado'.",
            cfg.ollama_model,
        )

    # -- 4. Teto de referência ----------------------------------------------
    if incluir_openai and cfg.openai_configurada:
        try:
            with etapa("avaliar:openai"):
                llm = obter_llm(cfg, origem="avaliacao.openai", temperatura=0.0, provedor="openai")
                resultados.append(
                    avaliar_sistema(
                        f"{cfg.openai_model} (teto de referência)", llm, casos[:n_openai]
                    )
                )
        except (ProvedorIndisponivelError, OrcamentoExcedidoError) as exc:
            log.warning("Teto de referência indisponível: %s", exc)
    elif incluir_openai:
        log.info("OPENAI_API_KEY não configurada — teto de referência pulado.")

    # -- Consolidação --------------------------------------------------------
    tabela = tabela_comparativa(resultados)
    log.info("\n%s", tabela)

    consolidado = {
        "conjunto_de_teste": {
            "arquivo": "data/processed/pubmedqa_teste.jsonl",
            "casos_avaliados": len(casos),
            "casos_disponiveis": len(carregar_teste(cfg)),
            "distribuicao": {
                r: sum(1 for c in casos if c["decisao"] == r) for r in ("yes", "no", "maybe")
            },
        },
        "referencia_externa": {
            "especialista_humano": 0.78,
            "fonte": "Jin et al., 2019 — artigo original do PubMedQA",
        },
        "sistemas": [r.para_dict(incluir_respostas=True) for r in resultados],
        "tabela": tabela,
        "custo": contador(cfg).por_modelo(),
        "custo_total_usd": round(contador(cfg).total_usd, 6),
    }

    destino = cfg.dir_docs / "avaliacao_resultados.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(consolidado, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Resultados gravados em %s", destino)

    contador(cfg).salvar(cfg)
    return {"consolidado": consolidado, "resultados": resultados}
