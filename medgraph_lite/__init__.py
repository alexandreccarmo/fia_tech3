"""
MedGraph Lite - assistente clinico auditavel.

Tech Challenge Fase 3 - 8IADT. Versao enxuta, desenhada para rodar inteira em
um Colab, em menos de 30 minutos, e ser demonstrada ao vivo.

Modulos:
    dados       PubMedQA, protocolos sinteticos e anonimizacao
    prontuario  base SQLite de pacientes
    treino      configuracao do fine-tuning QLoRA
    rag         indice FAISS e recuperacao com fonte
    guardrails  limites de atuacao e regras clinicas
    grafo       fluxo LangGraph de nove nos
    graficos    as cinco figuras da apresentacao
"""

__version__ = "1.0.0"
