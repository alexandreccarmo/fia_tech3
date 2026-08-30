"""
Recuperacao de evidencia com fonte rastreavel.

O item 3 do enunciado pede explainability: "indicar a fonte da informacao
utilizada na resposta". Isso so funciona se o trecho recuperado ja chegar
etiquetado ao prompt - o modelo cita o marcador que recebeu, e a verificacao
confere se aquele marcador existe.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@dataclass
class Trecho:
    marcador: str
    titulo: str
    texto: str


def montar_indice(protocolos: list[dict], embeddings):
    """
    Indexa os protocolos do hospital para busca por significado.

    FAISS roda em memoria e nao precisa de servico externo - a mesma escolha
    dos exemplos da aula.
    """
    documentos = [
        Document(
            page_content=f"{p['titulo']}. {p['texto']}",
            metadata={"marcador": p["id"], "titulo": p["titulo"]},
        )
        for p in protocolos
    ]
    return FAISS.from_documents(documentos, embeddings)


def recuperar(indice, pergunta: str, k: int = 2) -> list[Trecho]:
    return [
        Trecho(
            marcador=d.metadata["marcador"],
            titulo=d.metadata["titulo"],
            texto=d.page_content,
        )
        for d in indice.similarity_search(pergunta, k=k)
    ]


def montar_contexto(trechos: list[Trecho]) -> str:
    """Formata os trechos com o marcador visivel, que e o que o modelo cita."""
    return "\n\n".join(f"[{t.marcador}] {t.texto}" for t in trechos)
