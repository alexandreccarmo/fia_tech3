"""
Fixtures compartilhadas pelos testes do MedGraph.

DECISAO IMPORTANTE:
    Nenhum teste pode escrever em logs/ ou data/ do repositorio real. Um
    teste que suja o diretorio do projeto polui a trilha de auditoria - e a
    trilha de auditoria e justamente um dos entregaveis avaliados.

    A fixture `cfg_temporario` redireciona a raiz do projeto para um
    diretorio temporario do pytest. Isso funciona porque os caminhos em
    Settings sao propriedades calculadas a partir de `RAIZ_PROJETO` no
    momento do acesso, e nao constantes congeladas na criacao do objeto.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def cfg_temporario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Uma configuracao apontando para um diretorio temporario e descartavel."""
    import config.settings as modulo_settings
    from config.settings import Settings

    monkeypatch.setattr(modulo_settings, "RAIZ_PROJETO", tmp_path)

    cfg = Settings(
        _env_file=None,  # ignora o .env real da maquina
        llm_provider="eco",
        openai_api_key="",
        max_custo_usd_sessao=1.0,
        log_console_rich=False,
    )
    cfg.criar_diretorios()
    return cfg


@pytest.fixture
def logging_temporario(cfg_temporario, monkeypatch: pytest.MonkeyPatch):
    """Liga o logging apontando para o diretorio temporario e devolve a config."""
    import medgraph.logging_config as modulo_log

    monkeypatch.setattr(modulo_log, "_JA_CONFIGURADO", False)
    modulo_log.configurar_logging(cfg_temporario, forcar=True)
    yield cfg_temporario

    # Fecha os handlers para nao deixar arquivos abertos entre testes.
    import logging

    logger = logging.getLogger(modulo_log.LOGGER_RAIZ)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
