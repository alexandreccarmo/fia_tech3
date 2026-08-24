"""
[REQ-3c] Recuperação de evidência com rastreabilidade de fonte.

O QUE FAZ:
    Busca no índice vetorial os trechos mais relevantes para uma pergunta e os
    devolve já com o MARCADOR de citação atribuído — [E1], [P2], [C1] — pronto
    para ser injetado no prompt e conferido no guardrail de saída.

POR QUE O MARCADOR É ATRIBUÍDO AQUI, E NÃO NO PROMPT:
    O marcador é a chave que liga a afirmação do modelo ao texto que a
    sustenta. Se fosse gerado na montagem do prompt, o número dependeria da
    ordem de concatenação e mudaria entre execuções; a citação [E1] de uma
    resposta não corresponderia à [E1] guardada na trilha de auditoria, e a
    rastreabilidade — que é o requisito — se perderia.

    Atribuindo no recuperador, o marcador nasce junto do trecho, viaja com ele
    pelo estado do grafo, é gravado na auditoria e é resolvido de volta pelo
    painel. É um identificador estável de ponta a ponta.

A NUMERAÇÃO É POR TIPO, E NÃO GLOBAL:
    Os trechos de evidência recebem E1, E2, E3...; os de protocolo, P1, P2...;
    os do prontuário, C1, C2... Assim o médico sabe, só de ler a citação, se a
    afirmação vem da literatura, da norma interna do hospital ou do paciente
    à sua frente. São três níveis de autoridade diferentes, e confundi-los em
    uma numeração única esconderia a distinção.

SOBRE O ESCORE:
    O FAISS devolve DISTÂNCIA (menor é mais parecido), não similaridade. Como
    normalizamos os embeddings, a distância L2 ao quadrado se relaciona com o
    cosseno por `similaridade = 1 - distancia/2`. Convertemos aqui para que o
    painel exiba um número em que "maior é melhor", que é o que qualquer
    leitor espera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.logging_config import obter_logger
from medgraph.rag import indexar

log = obter_logger(__name__)

TipoFonte = Literal["evidencia", "protocolo", "clinico"]

# Prefixo do marcador conforme o tipo de fonte.
PREFIXO: dict[str, str] = {"evidencia": "E", "protocolo": "P", "clinico": "C"}

# Abaixo deste escore de similaridade, o trecho é considerado ruído e
# descartado. O valor foi calibrado observando as buscas dos protocolos: acima
# de 0,70 os trechos são pertinentes; entre 0,60 e 0,70 há mistura; abaixo
# disso, raramente há relação com a pergunta. Descartar é melhor do que
# entregar ruído: o modelo é instruído a usar APENAS o contexto, então um
# trecho irrelevante vira uma resposta errada com citação.
ESCORE_MINIMO = 0.62


@dataclass
class Trecho:
    """Um trecho recuperado, com tudo o que a citação precisa."""

    marcador: str
    """Identificador de citação: E1, P2, C1..."""

    tipo: TipoFonte
    titulo: str
    texto: str
    escore: float
    identificador: str = ""
    secao: str = ""
    metadados: dict[str, Any] = field(default_factory=dict)

    @property
    def rotulo_fonte(self) -> str:
        """Descrição legível da origem, usada no painel e nas citações."""
        nomes = {
            "evidencia": "Evidência científica",
            "protocolo": "Protocolo interno",
            "clinico": "Prontuário do paciente",
        }
        base = f"{nomes.get(self.tipo, self.tipo)} · {self.identificador}"
        return f"{base} · {self.secao}" if self.secao and self.secao != "abstract" else base

    def para_prompt(self) -> dict[str, str]:
        """Formato aceito por prompts.montar_contexto()."""
        return {
            "marcador": self.marcador,
            "titulo": f"{self.rotulo_fonte} — {self.titulo}",
            "texto": self.texto,
        }

    def para_dict(self) -> dict[str, Any]:
        return {
            "marcador": self.marcador,
            "tipo": self.tipo,
            "identificador": self.identificador,
            "titulo": self.titulo,
            "secao": self.secao,
            "escore": round(self.escore, 4),
            "texto": self.texto,
        }


class IndiceIndisponivelError(RuntimeError):
    """O índice vetorial não foi construído ainda."""


@lru_cache(maxsize=1)
def _carregar_indice(caminho: str, provedor: str, modelo: str):
    """
    Carrega o FAISS uma única vez por processo.

    O cache é essencial: carregar o índice envolve inicializar o modelo de
    embeddings, o que leva alguns segundos. Sem cache, cada consulta do grafo
    pagaria esse custo, e a latência do nó de recuperação seria dominada por
    inicialização em vez de busca.

    A chave inclui provedor e modelo para que trocar o embedding no .env
    invalide o cache em vez de misturar espaços vetoriais incompatíveis.
    """
    from langchain_community.vectorstores import FAISS

    embeddings = indexar.obter_embeddings()
    return FAISS.load_local(caminho, embeddings, allow_dangerous_deserialization=True)


class Recuperador:
    """Busca semântica sobre protocolos internos e evidência científica."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or obter_settings()
        caminho = indexar.caminho_indice(self.cfg)
        if not (caminho / "index.faiss").exists():
            raise IndiceIndisponivelError(
                f"Índice não encontrado em {caminho}.\nPara construir:  make indexar"
            )
        self._indice = _carregar_indice(
            str(caminho),
            self.cfg.embedding_provider,
            self.cfg.embedding_model_local
            if self.cfg.embedding_provider == "local"
            else self.cfg.embedding_model_openai,
        )

    # -- busca ---------------------------------------------------------------
    def recuperar(
        self,
        pergunta: str,
        *,
        k: int | None = None,
        escore_minimo: float = ESCORE_MINIMO,
        tipos: Sequence[str] | None = None,
    ) -> list[Trecho]:
        """
        Busca os trechos mais relevantes e atribui os marcadores de citação.

        Args:
            k: quantos trechos devolver. Padrão: RAG_TOP_K do .env.
            escore_minimo: corte de relevância. Passe 0.0 para desativar.
            tipos: restringe a origem, ex.: ["protocolo"]. Usado quando a
                intenção da pergunta é claramente sobre norma interna.

        Returns:
            Trechos ordenados por relevância, com marcadores já numerados.
        """
        k = k or self.cfg.rag_top_k

        # Buscamos mais do que o necessário porque o corte por escore e o
        # filtro por tipo vão descartar parte dos resultados.
        bruto = self._indice.similarity_search_with_score(pergunta, k=k * 3)

        candidatos: list[tuple[float, Any]] = []
        for documento, distancia in bruto:
            # Embeddings normalizados: converte distância L2² em similaridade
            # de cosseno, para que "maior é melhor".
            similaridade = max(0.0, 1.0 - float(distancia) / 2.0)
            if similaridade < escore_minimo:
                continue
            if tipos and documento.metadata.get("tipo") not in tipos:
                continue
            candidatos.append((similaridade, documento))

        candidatos.sort(key=lambda par: par[0], reverse=True)
        candidatos = candidatos[:k]

        contadores: dict[str, int] = {}
        trechos: list[Trecho] = []
        for similaridade, documento in candidatos:
            tipo = documento.metadata.get("tipo", "evidencia")
            prefixo = PREFIXO.get(tipo, "E")
            contadores[prefixo] = contadores.get(prefixo, 0) + 1

            trechos.append(
                Trecho(
                    marcador=f"{prefixo}{contadores[prefixo]}",
                    tipo=tipo,  # type: ignore[arg-type]
                    titulo=documento.metadata.get("titulo", "(sem título)"),
                    texto=documento.page_content,
                    escore=similaridade,
                    identificador=documento.metadata.get("id", ""),
                    secao=documento.metadata.get("secao", ""),
                    metadados=dict(documento.metadata),
                )
            )

        registrar(
            TipoEvento.RECUPERACAO,
            f"{len(trechos)} trecho(s) recuperado(s)",
            pergunta=pergunta[:150],
            k_pedido=k,
            candidatos_avaliados=len(bruto),
            escore_minimo=escore_minimo,
            fontes=[t.marcador for t in trechos],
            escores=[round(t.escore, 3) for t in trechos],
        )

        if not trechos:
            log.warning(
                "Nenhum trecho acima do escore mínimo (%.2f) para: %r",
                escore_minimo, pergunta[:80],
            )
        return trechos

    # -- utilidades ----------------------------------------------------------
    def estatisticas(self) -> dict[str, Any]:
        caminho = indexar.caminho_indice(self.cfg) / "estatisticas.json"
        if caminho.exists():
            import json

            return json.loads(caminho.read_text(encoding="utf-8"))
        return {"total_trechos": self._indice.index.ntotal}


def montar_contexto_para_prompt(trechos: Sequence[Trecho]) -> str:
    """Formata os trechos no bloco de contexto que o modelo lê."""
    from medgraph.chains import prompts

    return prompts.montar_contexto([t.para_prompt() for t in trechos])
