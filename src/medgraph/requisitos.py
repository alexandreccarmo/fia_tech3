"""
Catalogo dos requisitos do Tech Challenge - Fase 3 (8IADT).

O QUE FAZ:
    Transforma as exigencias escritas no PDF do Tech Challenge em um catalogo
    consultavel por codigo. Cada requisito ganha um identificador curto
    (ex.: "REQ-3b") que e usado como TAG nas docstrings de todo o projeto.

POR QUE EXISTE:
    O enunciado cobra um projeto que atenda a uma lista especifica de itens.
    Espalhar essas exigencias apenas em comentarios soltos torna dificil
    provar que todas foram cumpridas. Com este catalogo:

      1. Toda funcao relevante declara na docstring qual requisito atende,
         no formato [REQ-xx];
      2. O script scripts/gerar_rastreabilidade.py varre o codigo procurando
         essas tags e gera automaticamente docs/rastreabilidade.md, uma tabela
         "exigencia do PDF -> arquivo:linha";
      3. O professor consegue conferir a cobertura em poucos minutos, sem
         precisar ler o projeto inteiro.

COMO LER AS TAGS:
    Uma docstring que comeca com "[REQ-3b][REQ-3c]" significa que aquele
    trecho de codigo e responsavel por atender esses dois itens do enunciado.

Referencia: "8IADT - Fase 3 - Tech challenge.pdf", paginas 2 a 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Requisito:
    """Uma exigencia individual extraida do enunciado do Tech Challenge."""

    codigo: str
    """Identificador curto usado nas tags das docstrings. Ex.: 'REQ-3b'."""

    categoria: str
    """Bloco do enunciado ao qual o item pertence. Ex.: 'Seguranca e validacao'."""

    descricao: str
    """Texto do requisito, o mais proximo possivel da redacao original do PDF."""

    origem: str
    """Onde o requisito aparece no PDF, para conferencia rapida."""

    obrigatorio: bool = True
    """False apenas para itens que o enunciado apresenta como sugestao."""


# -----------------------------------------------------------------------------
# CATALOGO
# -----------------------------------------------------------------------------
# A ordem segue a leitura do PDF: primeiro os "Requisitos obrigatorios"
# (itens 1 a 4) e depois os "Entregaveis da Fase 3".
# -----------------------------------------------------------------------------
CATALOGO: Final[tuple[Requisito, ...]] = (
    # --- 1. Fine-tuning de LLM com dados medicos internos --------------------
    Requisito(
        codigo="REQ-1",
        categoria="Fine-tuning",
        descricao=(
            "Realizar o fine-tuning de um modelo LLM (como LLaMA, Falcon ou "
            "outro) utilizando protocolos medicos do hospital, exemplos de "
            "perguntas frequentes feitas por medicos e modelos de laudos, "
            "receitas e procedimentos internos."
        ),
        origem="PDF pag. 2, item 1",
    ),
    Requisito(
        codigo="REQ-1a",
        categoria="Fine-tuning",
        descricao=(
            "Preparar os dados com tecnicas de preprocessing, anonimizacao e "
            "curadoria."
        ),
        origem="PDF pag. 2, item 1, segundo marcador",
    ),
    # --- 2. Criacao de assistente medico com LangChain -----------------------
    Requisito(
        codigo="REQ-2",
        categoria="Assistente LangChain",
        descricao=(
            "Utilizar o LangChain para construir um pipeline que integre a "
            "LLM customizada."
        ),
        origem="PDF pag. 3, item 2",
    ),
    Requisito(
        codigo="REQ-2a",
        categoria="Assistente LangChain",
        descricao=(
            "Realizar consultas em base de dados estruturadas (como "
            "prontuarios e registros)."
        ),
        origem="PDF pag. 3, item 2",
    ),
    Requisito(
        codigo="REQ-2b",
        categoria="Assistente LangChain",
        descricao=(
            "Contextualizar as respostas da LLM com informacoes atualizadas "
            "do paciente."
        ),
        origem="PDF pag. 3, item 2",
    ),
    # --- 3. Seguranca e validacao --------------------------------------------
    Requisito(
        codigo="REQ-3a",
        categoria="Seguranca e validacao",
        descricao=(
            "Definir limites de atuacao do assistente para evitar sugestoes "
            "improprias (ex.: nunca prescrever diretamente, sem validacao "
            "humana)."
        ),
        origem="PDF pag. 3, item 3 (trecho destacado no enunciado)",
    ),
    Requisito(
        codigo="REQ-3b",
        categoria="Seguranca e validacao",
        descricao="Implementar logging detalhado para rastreamento e auditoria.",
        origem="PDF pag. 3, item 3",
    ),
    Requisito(
        codigo="REQ-3c",
        categoria="Seguranca e validacao",
        descricao=(
            "Garantir explainability das respostas da LLM (exemplo: indicar a "
            "fonte da informacao utilizada na resposta)."
        ),
        origem="PDF pag. 3, item 3",
    ),
    # --- 4. Organizacao do codigo --------------------------------------------
    Requisito(
        codigo="REQ-4",
        categoria="Organizacao do codigo",
        descricao="Projeto modularizado em Python com instrucoes completas no README.",
        origem="PDF pag. 3, item 4",
    ),
    # --- Entregaveis da Fase 3 -----------------------------------------------
    Requisito(
        codigo="REQ-E1",
        categoria="Entregaveis",
        descricao="Codigo-fonte com os fluxos do LangGraph.",
        origem="PDF pag. 3, Entregaveis / Repositorio Git",
    ),
    Requisito(
        codigo="REQ-E2",
        categoria="Entregaveis",
        descricao="Dataset anonimizado ou exemplo de dados sinteticos.",
        origem="PDF pag. 3, Entregaveis / Repositorio Git",
    ),
    Requisito(
        codigo="REQ-E3",
        categoria="Entregaveis",
        descricao=(
            "Relatorio tecnico detalhado com explicacao do processo de "
            "fine-tuning, descricao do assistente medico criado, diagrama do "
            "fluxo LangChain e avaliacao do modelo com analise dos resultados."
        ),
        origem="PDF pag. 3-4, Entregaveis",
    ),
    Requisito(
        codigo="REQ-E4",
        categoria="Entregaveis",
        descricao=(
            "Video de ate 15 minutos demonstrando o treinamento e "
            "funcionamento da LLM personalizada, a execucao de um fluxo "
            "automatizado, respostas a perguntas clinicas contextualizadas e "
            "os logs e a validacao das respostas."
        ),
        origem="PDF pag. 4",
    ),
)


# Indice por codigo, para consulta O(1) a partir de uma tag encontrada no codigo.
POR_CODIGO: Final[dict[str, Requisito]] = {r.codigo: r for r in CATALOGO}


def obter(codigo: str) -> Requisito:
    """
    Devolve o requisito correspondente a uma tag.

    Levanta KeyError com mensagem explicita quando a tag nao existe - isso
    protege contra erros de digitacao nas docstrings (um "REQ-3d" que nao
    existe, por exemplo), que passariam despercebidos e sumiriam do relatorio
    de rastreabilidade.

    Observacao: o codigo de exemplo acima aparece SEM colchetes de proposito.
    Escrito na forma de tag, ele seria capturado pelo proprio gerador da
    matriz de rastreabilidade e reportado como tag invalida.
    """
    try:
        return POR_CODIGO[codigo]
    except KeyError as exc:
        conhecidos = ", ".join(sorted(POR_CODIGO))
        raise KeyError(
            f"Requisito '{codigo}' nao existe no catalogo. Codigos validos: {conhecidos}"
        ) from exc


def listar_categorias() -> list[str]:
    """Categorias na ordem em que aparecem no PDF, sem repeticao."""
    vistas: list[str] = []
    for requisito in CATALOGO:
        if requisito.categoria not in vistas:
            vistas.append(requisito.categoria)
    return vistas
