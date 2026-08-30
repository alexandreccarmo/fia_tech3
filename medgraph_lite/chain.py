"""
O pipeline LangChain que integra a LLM ajustada.

O item 2 do enunciado pede, literalmente, "utilizar o LangChain para construir
um pipeline que integre a LLM customizada". Chamar `modelo.generate()` direto
produziria a mesma resposta, mas nao seria isso - e a diferenca nao e
burocratica:

    o prompt vira um objeto versionavel, e nao uma f-string espalhada;
    a cadeia e composta com `|`, o mesmo operador dos exemplos da aula;
    trocar a LLM (ajustada, base, ou uma API) nao mexe no resto do fluxo.

A cadeia montada aqui e:

    ChatPromptTemplate  ->  HuggingFacePipeline  ->  StrOutputParser
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SISTEMA = (
    "Voce e um assistente clinico do Hospital Vida Plena. Responda SEMPRE neste "
    "formato:\n"
    "Decisao: yes|no|maybe\n"
    "<justificativa em ate 3 frases, apoiada apenas no contexto fornecido>\n"
    "Cite a fonte entre colchetes, por exemplo [P1] ou [E1].\n"
    "Nunca prescreva: apresente a evidencia e devolva a decisao ao medico."
)

USUARIO = "Contexto:\n{contexto}\n\nPergunta: {pergunta}"

PROMPT = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", USUARIO)])


def montar_llm(modelo, tokenizador, max_novos_tokens: int = 140):
    """
    Embrulha o modelo ajustado num componente LangChain.

    `HuggingFacePipeline` e a ponte entre um modelo carregado em memoria e o
    resto do ecossistema: dali em diante ele se compoe como qualquer outra LLM.
    """
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    gerador = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizador,
        max_new_tokens=max_novos_tokens,
        do_sample=False,
        # Sem isto o pipeline devolve o prompt inteiro junto com a resposta, e
        # a verificacao de formato passaria a olhar o texto errado.
        return_full_text=False,
        pad_token_id=tokenizador.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=gerador)


def montar_cadeia(llm):
    """
    Compoe prompt, modelo e parser com o operador `|`.

    E a mesma forma dos exemplos da aula. O parser no fim garante que o resto do
    codigo receba texto, e nao o objeto de mensagem da biblioteca.
    """
    return PROMPT | llm | StrOutputParser()


def responder(cadeia, pergunta: str, contexto: str) -> str:
    return cadeia.invoke({"pergunta": pergunta, "contexto": contexto}).strip()
