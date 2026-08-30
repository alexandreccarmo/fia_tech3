"""
[REQ-2][REQ-3c] Construção do índice vetorial.

O QUE FAZ:
    Fatia as duas bases de conhecimento — a evidência científica do PubMedQA e
    os protocolos internos do hospital — em trechos, calcula os embeddings e
    grava um índice FAISS em disco.

AS DUAS BASES E POR QUE FICAM NO MESMO ÍNDICE:
    Uma dúvida clínica real quase nunca se resolve só com um dos dois. "Qual
    antibiótico empírico para sepse de foco urinário?" precisa do protocolo
    interno (o que o hospital padronizou) e da evidência (o que a literatura
    sustenta). Índices separados obrigariam a decidir de antemão onde buscar;
    um índice único deixa a similaridade decidir, e o metadado de cada trecho
    preserva a distinção na hora de citar.

MARCADORES DE FONTE — o mecanismo de explainability  [REQ-3c]:
    Cada trecho carrega o tipo da sua origem, e o recuperador o converte em um
    marcador que o modelo cita na resposta:

      [E#]  evidência científica (abstract do PubMedQA)
      [P#]  protocolo interno do Hospital Vida Plena

    Esse metadado é o que permite ao painel resolver a citação de volta para o
    texto original e ao guardarail de saída verificar que toda afirmação tem
    procedência. Sem ele, a citação seria decorativa.

POR QUE EMBEDDINGS LOCAIS POR PADRÃO:
    `multilingual-e5-small` cobre inglês (os abstracts) e português (os
    protocolos) no mesmo espaço vetorial, roda em CPU, e custa zero. Durante o
    desenvolvimento o índice é reconstruído dezenas de vezes; com embeddings
    pagos, cada reconstrução teria preço. O provedor da OpenAI continua
    disponível por configuração.

Uso:
    make indexar
    python -m medgraph.rag.indexar
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

NOME_INDICE = "faiss_medgraph"

# Quantos abstracts do PubMedQA entram na base de evidência. Vêm do conjunto
# de TREINO e do artificial — nunca do conjunto de teste, que precisa
# permanecer inédito para a avaliação.
MAX_ABSTRACTS_EVIDENCIA = 1200


@dataclass
class EstatisticasIndice:
    """Retrato do índice construído. Vai para o relatório e para o painel."""

    total_trechos: int = 0
    trechos_por_fonte: dict[str, int] | None = None
    documentos_originais: int = 0
    caracteres_totais: int = 0
    modelo_embedding: str = ""
    dimensao: int = 0
    duracao_s: float = 0.0

    def para_dict(self) -> dict[str, Any]:
        return {
            "total_trechos": self.total_trechos,
            "trechos_por_fonte": self.trechos_por_fonte or {},
            "documentos_originais": self.documentos_originais,
            "caracteres_totais": self.caracteres_totais,
            "modelo_embedding": self.modelo_embedding,
            "dimensao": self.dimensao,
            "duracao_s": round(self.duracao_s, 1),
        }


# =============================================================================
# EMBEDDINGS
# =============================================================================
class EmbeddingsComPrefixo(Embeddings):
    """
    Adiciona os prefixos de instrução exigidos pelos modelos da família E5.

    POR QUE ISSO PRECISA EXISTIR:
        Os modelos `intfloat/e5` foram treinados de forma assimétrica: a
        pergunta é apresentada como `query: <texto>` e o documento como
        `passage: <texto>`. Os dois prefixos colocam pergunta e documento em
        regiões diferentes do espaço vetorial, e é essa assimetria que faz a
        recuperação funcionar bem.

        Omitir os prefixos NÃO gera erro. As buscas continuam retornando
        resultados — apenas piores. É exatamente o tipo de degradação
        silenciosa que passa despercebida até alguém investigar por que o RAG
        recupera trechos vagamente relacionados.

        A classe `HuggingFaceEmbeddings` do LangChain não expõe campos para
        esses prefixos (os campos `query_instruction` e `embed_instruction`
        pertencem ao wrapper específico de modelos BGE). Envolvê-la é a forma
        correta e explícita de resolver.

    A distinção entre `embed_documents` e `embed_query` é justamente onde os
    dois prefixos se separam — e é por isso que a interface do LangChain tem
    dois métodos em vez de um.
    """

    def __init__(self, base: Embeddings, prefixo_documento: str, prefixo_consulta: str) -> None:
        self._base = base
        self._prefixo_documento = prefixo_documento
        self._prefixo_consulta = prefixo_consulta

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([self._prefixo_documento + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._prefixo_consulta + text)


def obter_embeddings(cfg: Settings | None = None) -> Embeddings:
    """Devolve o modelo de embeddings configurado, com os prefixos corretos."""
    cfg = cfg or obter_settings()

    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not cfg.openai_configurada:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai, mas OPENAI_API_KEY está vazia no .env."
            )
        return OpenAIEmbeddings(model=cfg.embedding_model_openai, api_key=cfg.openai_api_key)

    # Usamos a implementação de langchain_community de propósito. O pacote
    # dedicado `langchain-huggingface` exigiria langchain-core 1.x, o que
    # quebraria a compatibilidade com o restante do stack (langchain 0.3.x,
    # langchain-openai, langgraph). Como o wrapper acima já resolve os
    # prefixos — o único motivo pelo qual o pacote dedicado seria útil —,
    # a dependência extra não se justifica.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    base = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model_local,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if "e5" in cfg.embedding_model_local.lower():
        return EmbeddingsComPrefixo(base, "passage: ", "query: ")
    return base


# =============================================================================
# CARREGAMENTO DAS FONTES
# =============================================================================
def _carregar_protocolos(cfg: Settings) -> list[Document]:
    """
    Carrega os protocolos internos, fatiando por SEÇÃO.

    POR QUE FATIAR POR SEÇÃO E NÃO POR TAMANHO FIXO:
        Um protocolo clínico tem estrutura semântica: "Conduta terapêutica" é
        uma unidade de sentido, "Alertas e contraindicações" é outra. Um corte
        por número de caracteres partiria uma tabela de doses ao meio e
        devolveria metade dela ao modelo — que apresentaria a dose incompleta
        como se fosse completa. Aqui o corte respeita a fronteira da seção, e
        só então, se a seção for longa demais, o divisor por tamanho atua
        dentro dela.
    """
    pasta = cfg.dir_dados_sinteticos / "protocolos"
    if not pasta.is_dir():
        log.warning("Pasta de protocolos não encontrada: %s", pasta)
        return []

    import re

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag_chunk_size,
        chunk_overlap=cfg.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documentos: list[Document] = []
    for arquivo in sorted(pasta.glob("PROT-*.md")):
        texto = arquivo.read_text(encoding="utf-8")

        titulo_m = re.search(r"^titulo:\s*(.+)$", texto, re.MULTILINE)
        titulo = titulo_m.group(1).strip() if titulo_m else arquivo.stem
        identificador = "-".join(arquivo.name.split("-")[:2])

        corpo = texto.split("---", 2)[-1] if texto.startswith("---") else texto

        for bloco in re.split(r"\n(?=## )", corpo):
            bloco = bloco.strip()
            # A seção de aviso e a de referências não têm conteúdo clínico
            # recuperável; indexá-las só geraria ruído nas buscas.
            if len(bloco) < 120 or bloco.startswith(("## Aviso", "## 9. Refer")):
                continue

            cabecalho_m = re.match(r"##\s*(.+)", bloco)
            secao = cabecalho_m.group(1).strip() if cabecalho_m else "(sem seção)"

            for parte in divisor.split_text(bloco):
                documentos.append(
                    Document(
                        page_content=parte,
                        metadata={
                            "tipo": "protocolo",
                            "id": identificador,
                            "titulo": titulo,
                            "secao": secao,
                            "arquivo": arquivo.name,
                        },
                    )
                )
    return documentos


def _carregar_evidencia(cfg: Settings) -> list[Document]:
    """
    Carrega os abstracts do PubMedQA como base de evidência científica.

    RESTRIÇÃO IMPORTANTE:
        Só entram registros do conjunto de TREINO e do artificial. Os 500
        exemplos de teste ficam de fora do índice — se estivessem lá, o
        assistente poderia recuperar o próprio abstract da pergunta que está
        sendo avaliada, e a avaliação da Etapa 4 mediria recuperação em vez de
        raciocínio.
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag_chunk_size,
        chunk_overlap=cfg.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documentos: list[Document] = []
    vistos: set[Any] = set()

    for arquivo in ("pubmedqa_treino.jsonl", "pubmedqa_artificial.jsonl"):
        caminho = cfg.dir_dados_processados / arquivo
        if not caminho.exists():
            continue

        with caminho.open(encoding="utf-8") as fonte:
            for linha in fonte:
                if len(vistos) >= MAX_ABSTRACTS_EVIDENCIA:
                    break
                linha = linha.strip()
                if not linha:
                    continue

                registro = json.loads(linha)
                pubid = registro.get("pubid")
                if pubid in vistos:
                    continue
                vistos.add(pubid)

                for parte in divisor.split_text(registro["contexto"]):
                    documentos.append(
                        Document(
                            page_content=parte,
                            metadata={
                                "tipo": "evidencia",
                                "id": f"pubmed:{pubid}",
                                # A pergunta de pesquisa é o título mais
                                # informativo que temos para um abstract.
                                "titulo": registro["pergunta"][:150],
                                "secao": "abstract",
                                "decisao": registro.get("decisao", ""),
                            },
                        )
                    )
    return documentos


# =============================================================================
# CONSTRUÇÃO
# =============================================================================
def construir(cfg: Settings | None = None, *, forcar: bool = False) -> EstatisticasIndice:
    """Monta o índice FAISS e o grava em data/indices/faiss_medgraph."""
    cfg = cfg or obter_settings()
    destino = cfg.dir_indices / NOME_INDICE

    if destino.exists() and not forcar:
        log.info("Índice já existe em %s — use forcar=True para reconstruir.", destino)
        caminho_estatisticas = destino / "estatisticas.json"
        if caminho_estatisticas.exists():
            dados = json.loads(caminho_estatisticas.read_text(encoding="utf-8"))
            return EstatisticasIndice(**{
                k: v for k, v in dados.items()
                if k in EstatisticasIndice.__dataclass_fields__
            })
        return EstatisticasIndice()

    inicio = time.perf_counter()

    with etapa("indexar:carregar_fontes", tipo=TipoEvento.RECUPERACAO):
        protocolos = _carregar_protocolos(cfg)
        evidencia = _carregar_evidencia(cfg)
        documentos = protocolos + evidencia

        if not documentos:
            raise RuntimeError(
                "Nenhum documento para indexar. Rode primeiro: make dados"
            )

        log.info(
            "Trechos: %d de protocolos internos + %d de evidência científica = %d",
            len(protocolos), len(evidencia), len(documentos),
        )

    with etapa("indexar:embeddings", tipo=TipoEvento.RECUPERACAO):
        embeddings = obter_embeddings(cfg)
        log.info(
            "Calculando embeddings com %s (a primeira execução baixa o modelo)...",
            cfg.embedding_model_local if cfg.embedding_provider == "local"
            else cfg.embedding_model_openai,
        )

        from langchain_community.vectorstores import FAISS

        indice = FAISS.from_documents(documentos, embeddings)

    with etapa("indexar:gravar", tipo=TipoEvento.RECUPERACAO):
        if destino.exists():
            shutil.rmtree(destino)
        destino.mkdir(parents=True, exist_ok=True)
        indice.save_local(str(destino))

    estatisticas = EstatisticasIndice(
        total_trechos=len(documentos),
        trechos_por_fonte={
            "protocolo interno": len(protocolos),
            "evidência científica": len(evidencia),
        },
        documentos_originais=len({d.metadata["id"] for d in documentos}),
        caracteres_totais=sum(len(d.page_content) for d in documentos),
        modelo_embedding=(
            cfg.embedding_model_local if cfg.embedding_provider == "local"
            else cfg.embedding_model_openai
        ),
        dimensao=indice.index.d,
        duracao_s=time.perf_counter() - inicio,
    )
    (destino / "estatisticas.json").write_text(
        json.dumps(estatisticas.para_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    registrar(
        TipoEvento.RECUPERACAO,
        "Índice vetorial construído",
        etapa="indexar",
        conclusao=True,
        **estatisticas.para_dict(),
    )
    log.info(
        "Índice gravado em %s — %d trechos, dimensão %d, %.1f s",
        destino, estatisticas.total_trechos, estatisticas.dimensao, estatisticas.duracao_s,
    )
    return estatisticas


def caminho_indice(cfg: Settings | None = None) -> Path:
    cfg = cfg or obter_settings()
    return cfg.dir_indices / NOME_INDICE


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Índice vetorial", subtitulo="protocolos internos + evidência científica")
    construir(forcar=True)
