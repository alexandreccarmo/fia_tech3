"""
[REQ-4] Configuracao centralizada do MedGraph.

O QUE FAZ:
    Le todas as variaveis de ambiente do arquivo .env, valida os valores e
    expoe um unico objeto `Settings` que o projeto inteiro consulta. Tambem
    calcula os caminhos absolutos das pastas de dados, logs, modelos e
    documentacao a partir da raiz do repositorio.

POR QUE EXISTE:
    Sem isso, cada modulo faria seu proprio `os.getenv(...)` com um default
    diferente, e descobrir "qual modelo esta sendo usado" exigiria caçar
    string por string. Concentrar a configuracao em um lugar so atende ao
    requisito de projeto modularizado (item 4 do enunciado) e evita que
    segredos vazem para o codigo-fonte.

COMO USAR:
    from config.settings import obter_settings
    cfg = obter_settings()
    print(cfg.llm_provider, cfg.dir_logs)

DECISOES DE PROJETO:
    - `obter_settings()` e cacheado: a configuracao e lida do disco uma unica
      vez por processo, entao todos os modulos enxergam exatamente os mesmos
      valores durante uma execucao.
    - Nenhum default aponta para servico pago. O padrao e o modelo local
      (Ollama); a OpenAI so entra quando explicitamente configurada.
    - Se a OPENAI_API_KEY estiver vazia, isso NAO e erro: o projeto foi
      desenhado para rodar 100% offline. O erro so aparece se alguem tentar
      efetivamente usar o provider `openai` sem chave.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# RAIZ DO REPOSITORIO
# -----------------------------------------------------------------------------
# Este arquivo mora em <raiz>/config/settings.py, entao dois niveis acima
# chegamos na raiz. Usar caminho relativo ao arquivo (e nao ao diretorio de
# trabalho) faz os scripts funcionarem de qualquer lugar que sejam chamados.
RAIZ_PROJETO: Path = Path(__file__).resolve().parent.parent


ProvedorLLM = Literal["ollama", "openai", "eco"]
ProvedorEmbedding = Literal["local", "openai"]


class Settings(BaseSettings):
    """Configuracao do MedGraph, carregada de variaveis de ambiente e do .env."""

    model_config = SettingsConfigDict(
        env_file=RAIZ_PROJETO / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # variaveis extras no .env nao quebram a aplicacao
    )

    # -------------------------------------------------------------------------
    # IDENTIDADE DO PROJETO
    # -------------------------------------------------------------------------
    nome_projeto: str = "MedGraph"
    versao: str = "0.1.0"
    hospital: str = "Hospital Vida Plena"

    # -------------------------------------------------------------------------
    # 1) PROVEDOR DE LLM
    # -------------------------------------------------------------------------
    llm_provider: ProvedorLLM = Field(
        default="ollama",
        description=(
            "ollama = modelo fine-tunado local (padrao) | "
            "openai = gpt-4o-mini (teto de referencia) | "
            "eco = heuristica offline, sem LLM"
        ),
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "medgraph"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_s: int = Field(default=120, gt=0)

    # -------------------------------------------------------------------------
    # 2) CONTROLE DE CUSTO  [REQ-3b]
    # -------------------------------------------------------------------------
    max_custo_usd_sessao: float = Field(
        default=1.00,
        ge=0.0,
        description="Teto de gasto por execucao. 0.0 bloqueia qualquer chamada paga.",
    )
    habilitar_cache_llm: bool = True

    # -------------------------------------------------------------------------
    # 3) EMBEDDINGS E RAG
    # -------------------------------------------------------------------------
    embedding_provider: ProvedorEmbedding = "local"
    embedding_model_local: str = "intfloat/multilingual-e5-small"
    embedding_model_openai: str = "text-embedding-3-small"

    rag_chunk_size: int = Field(default=900, gt=0)
    rag_chunk_overlap: int = Field(default=150, ge=0)
    rag_top_k: int = Field(default=4, gt=0)

    # -------------------------------------------------------------------------
    # 4) COMPORTAMENTO DO GRAFO  [REQ-3a]
    # -------------------------------------------------------------------------
    max_tentativas_guardrail: int = Field(default=2, ge=0, le=5)
    limiar_risco_validacao_humana: float = Field(default=0.6, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # 5) LOGGING E AUDITORIA  [REQ-3b]
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_console_rich: bool = True
    log_arquivo: bool = True
    log_auditoria_jsonl: bool = True
    log_salvar_trace_completo: bool = True

    # -------------------------------------------------------------------------
    # 6) FINE-TUNING E MODELO
    # -------------------------------------------------------------------------
    modelo_base_hf: str = "meta-llama/Llama-3.2-3B-Instruct"
    # Valor de EXEMPLO. O repositorio real e criado pelo usuario ao final do
    # notebook de exportacao, que imprime o nome a colocar no .env. Deixar um
    # nome plausivel aqui, em vez de vazio, mantem a mensagem de erro do
    # instalador legivel quando o modelo ajustado ainda nao existe.
    repo_gguf_hf: str = "seu-usuario/medgraph-llama32-3b-gguf"
    arquivo_gguf: str = "medgraph-llama32-3b-q4_k_m.gguf"
    hf_token: str = ""

    # -------------------------------------------------------------------------
    # 7) AMOSTRAGEM DO DATASET
    # -------------------------------------------------------------------------
    pubmedqa_artificial_amostra: int = Field(default=8000, ge=0)
    semente_aleatoria: int = 42

    # =========================================================================
    # VALIDADORES
    # =========================================================================
    @field_validator("log_level")
    @classmethod
    def _validar_log_level(cls, valor: str) -> str:
        """Aceita o nivel em qualquer caixa, mas normaliza para maiusculo."""
        niveis = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalizado = valor.strip().upper()
        if normalizado not in niveis:
            raise ValueError(
                f"LOG_LEVEL invalido: {valor!r}. Use um de: {', '.join(sorted(niveis))}"
            )
        return normalizado

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _validar_overlap(cls, valor: int, info) -> int:
        """
        A sobreposicao precisa ser menor que o tamanho do chunk.

        Se fosse maior ou igual, o splitter entraria em laco infinito ou
        produziria chunks duplicados - um bug silencioso que so apareceria
        na hora de indexar milhares de documentos.
        """
        chunk_size = info.data.get("rag_chunk_size")
        if chunk_size is not None and valor >= chunk_size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({valor}) deve ser menor que "
                f"RAG_CHUNK_SIZE ({chunk_size})."
            )
        return valor

    @field_validator("ollama_base_url")
    @classmethod
    def _normalizar_url(cls, valor: str) -> str:
        """Remove a barra final para evitar URLs com '//' ao concatenar rotas."""
        return valor.rstrip("/")

    # =========================================================================
    # CAMINHOS DERIVADOS
    # =========================================================================
    # Sao propriedades calculadas (nao variaveis de ambiente) para garantir que
    # a estrutura de pastas do projeto seja sempre a mesma, independente de
    # onde o script foi chamado.
    # =========================================================================
    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_raiz(self) -> Path:
        return RAIZ_PROJETO

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_dados(self) -> Path:
        return RAIZ_PROJETO / "data"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_dados_brutos(self) -> Path:
        """PubMedQA como veio do Hugging Face, sem nenhum tratamento."""
        return self.dir_dados / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_dados_processados(self) -> Path:
        """Datasets ja anonimizados, curados e prontos para o fine-tuning."""
        return self.dir_dados / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_dados_sinteticos(self) -> Path:
        """Corpus hospitalar gerado por nos: protocolos, FAQ, laudos, prontuarios."""
        return self.dir_dados / "sintetico"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_indices(self) -> Path:
        """Indices vetoriais FAISS persistidos."""
        return self.dir_dados / "indices"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def caminho_banco_prontuarios(self) -> Path:
        """Base SQLite com os prontuarios estruturados. [REQ-2a]"""
        return self.dir_dados_sinteticos / "prontuarios.sqlite"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_logs(self) -> Path:
        return RAIZ_PROJETO / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_auditoria(self) -> Path:
        """Trilha de auditoria em JSONL, um arquivo por dia. [REQ-3b]"""
        return self.dir_logs / "auditoria"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_traces(self) -> Path:
        """Trace completo de cada consulta, um JSON por trace_id. [REQ-3b]"""
        return self.dir_logs / "traces"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_modelos(self) -> Path:
        return RAIZ_PROJETO / "models"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_adapters(self) -> Path:
        """Adapters LoRA produzidos no Colab (versionados no Git)."""
        return self.dir_modelos / "adapters"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_gguf(self) -> Path:
        """Modelo quantizado baixado do Hugging Face Hub (fora do Git)."""
        return self.dir_modelos / "gguf"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_docs(self) -> Path:
        return RAIZ_PROJETO / "docs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_graficos(self) -> Path:
        """Saida dos graficos de avaliacao usados no relatorio tecnico."""
        return self.dir_docs / "graficos"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dir_diagramas(self) -> Path:
        """Diagramas do grafo LangGraph (ASCII, Mermaid e PNG)."""
        return self.dir_docs / "diagramas"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def caminho_politicas(self) -> Path:
        """Regras declarativas de guardrail. [REQ-3a]"""
        return RAIZ_PROJETO / "config" / "politicas.yaml"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def caminho_cache_llm(self) -> Path:
        return RAIZ_PROJETO / "cache_llm.sqlite"

    # =========================================================================
    # AUXILIARES
    # =========================================================================
    @property
    def openai_configurada(self) -> bool:
        """True quando ha chave da OpenAI disponivel para uso."""
        return bool(self.openai_api_key.strip())

    def criar_diretorios(self) -> None:
        """
        Garante que toda a arvore de pastas existe antes de qualquer escrita.

        Chamado no bootstrap da aplicacao. Evita a classe de erro mais chata
        do projeto: um pipeline rodar 40 minutos e falhar no final porque a
        pasta de saida nao existia.
        """
        for caminho in (
            self.dir_dados_brutos,
            self.dir_dados_processados,
            self.dir_dados_sinteticos,
            self.dir_indices,
            self.dir_logs,
            self.dir_auditoria,
            self.dir_traces,
            self.dir_adapters,
            self.dir_gguf,
            self.dir_graficos,
            self.dir_diagramas,
        ):
            caminho.mkdir(parents=True, exist_ok=True)

    def resumo_seguro(self) -> dict[str, object]:
        """
        Retrato da configuracao com os segredos mascarados.

        Este dicionario e gravado no inicio de toda execucao na trilha de
        auditoria, para que seja possivel reconstruir depois em que condicoes
        um resultado foi produzido - sem nunca escrever a chave da API em
        disco. [REQ-3b]
        """

        def _mascarar(segredo: str) -> str:
            if not segredo:
                return "(nao definido)"
            return f"{segredo[:7]}...{segredo[-4:]}" if len(segredo) > 14 else "(definido)"

        return {
            "projeto": self.nome_projeto,
            "versao": self.versao,
            "llm_provider": self.llm_provider,
            "ollama_model": self.ollama_model,
            "openai_model": self.openai_model,
            "openai_api_key": _mascarar(self.openai_api_key),
            "hf_token": _mascarar(self.hf_token),
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "embedding_provider": self.embedding_provider,
            "embedding_model": (
                self.embedding_model_local
                if self.embedding_provider == "local"
                else self.embedding_model_openai
            ),
            "rag_top_k": self.rag_top_k,
            "max_custo_usd_sessao": self.max_custo_usd_sessao,
            "max_tentativas_guardrail": self.max_tentativas_guardrail,
            "limiar_risco_validacao_humana": self.limiar_risco_validacao_humana,
            "modelo_base_hf": self.modelo_base_hf,
            "semente_aleatoria": self.semente_aleatoria,
        }


@lru_cache(maxsize=1)
def obter_settings() -> Settings:
    """
    Devolve a instancia unica de configuracao do processo.

    O cache do lru_cache garante que o .env seja lido uma unica vez. Em
    testes, use `obter_settings.cache_clear()` antes de alterar variaveis de
    ambiente para forcar a releitura.
    """
    return Settings()
