"""
MedGraph - Assistente Clinico Auditavel.

Tech Challenge Fase 3 - Pos-Tech 8IADT
Hospital Vida Plena (cenario ficticio)

Integrantes do grupo:
    Alexandre Carneiro do Carmo      - RM370980
    Brunno Costa Castigrini          - RM371429
    Pedro Henrique Azevedo Aragao    - RM373481
    Valter Willian de Oliveira Filho - RM370979

VISAO GERAL DO PACOTE:

    medgraph/
      requisitos.py     catalogo dos requisitos do enunciado (tags [REQ-xx])
      logging_config.py logging em tres destinos                    [REQ-3b]
      auditoria.py      trilha de auditoria e trace por consulta    [REQ-3b]
      dados/            download, anonimizacao e curadoria          [REQ-1a]
      finetune/         preparo do dataset e notebooks do Colab     [REQ-1]
      avaliacao/        metricas e graficos do relatorio            [REQ-E3]
      llm/              provedores de modelo e controle de custo    [REQ-2]
      rag/              indice vetorial e recuperacao com fontes    [REQ-3c]
      prontuario/       acesso a base estruturada de pacientes      [REQ-2a]
      chains/           pipelines LangChain                         [REQ-2]
      guardrails/       limites de atuacao do assistente            [REQ-3a]
      grafo/            fluxo LangGraph                             [REQ-E1]
      ui/               painel Streamlit
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "iniciar"]


def iniciar(*, banner: str | None = None, subtitulo: str = "") -> None:
    """
    Bootstrap padrao de qualquer script ou notebook do projeto.

    O QUE FAZ, NESTA ORDEM:
        1. Le a configuracao do .env e valida os valores;
        2. Cria toda a arvore de diretorios (data/, logs/, models/, docs/);
        3. Liga o logging nos tres destinos;
        4. Opcionalmente imprime o banner visual da etapa.

    POR QUE CENTRALIZAR:
        Sem isso, cada script repetiria as mesmas quatro chamadas - e o
        primeiro que esquecesse uma delas produziria uma execucao sem rastro
        de auditoria, justamente o que o requisito 3 do enunciado proibe.

    Uso:
        from medgraph import iniciar
        iniciar(banner="Etapa 1 - Preparacao dos dados")
    """
    from config.settings import obter_settings
    from medgraph.logging_config import configurar_logging, imprimir_banner

    cfg = obter_settings()
    cfg.criar_diretorios()
    configurar_logging(cfg)

    if banner:
        imprimir_banner(banner, subtitulo, cfg)
