"""
[REQ-2] Provedor de modelo de linguagem.

O QUE FAZ:
    Devolve um modelo de chat do LangChain conforme a configuracao, sempre
    com a mesma interface. O resto do projeto - chains, nos do grafo,
    avaliacao - nunca precisa saber qual motor esta por tras.

OS TRES PROVEDORES E POR QUE CADA UM EXISTE:

    ollama  (padrao)
        A LLM CUSTOMIZADA deste projeto: o Llama-3.2-3B ajustado por QLoRA,
        convertido para GGUF e servido localmente. E o provedor que atende ao
        requisito 2 do enunciado - "construir um pipeline que integre a LLM
        customizada". Custo zero por consulta, roda offline, e o codigo
        LangChain e o mesmo padrao visto na Aula 05.

    openai
        gpt-4o-mini. NAO e o assistente: serve como TETO DE REFERENCIA na
        avaliacao da Etapa 4 e como plano B quando o Ollama nao esta no ar.
        Toda chamada por aqui passa pela trava de orcamento.

    eco
        Um modelo falso, deterministico, sem rede. Existe por tres razoes
        concretas:
          1. os testes automatizados precisam exercitar o grafo inteiro sem
             depender de um servidor externo estar rodando;
          2. a demonstracao precisa funcionar mesmo se o Ollama falhar na
             hora da apresentacao;
          3. permite medir o custo REAL da orquestracao - quanto do tempo de
             resposta e LangGraph e quanto e inferencia.
        As respostas do eco respeitam o formato de saida (decisao na primeira
        linha, citacao no fim), entao os guardrails e o parser sao exercitados
        de verdade.

DECISAO DE PROJETO - por que devolver um objeto do LangChain, e nao um cliente proprio:
    Devolvendo um `BaseChatModel`, todo o resto do projeto usa a sintaxe de
    composicao que o curso ensinou: `prompt | llm | parser`. Um cliente HTTP
    proprio economizaria uma dependencia e custaria a integracao com todo o
    ecossistema - parsers, callbacks, streaming, retries.

Uso:
    from medgraph.llm.provider import obter_llm

    llm = obter_llm()
    resposta = (prompt | llm).invoke({"pergunta": "..."})
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.llm.custo import contador, estimar_tokens
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


class ProvedorIndisponivelError(RuntimeError):
    """
    O provedor configurado nao pode ser usado agora.

    Traz sempre a instrucao concreta de como resolver - "o Ollama nao
    respondeu" sem dizer o que fazer nao ajuda ninguem as onze da noite.
    """


# =============================================================================
# CALLBACK DE CUSTO  [REQ-3b]
# =============================================================================
class CallbackCusto(BaseCallbackHandler):
    """
    Contabiliza tokens e dolares de cada chamada, automaticamente.

    POR QUE UM CALLBACK E NAO UMA CHAMADA EXPLICITA:
        Contabilizar manualmente exigiria lembrar de fazer isso em cada ponto
        do projeto que invoca o modelo - e o primeiro esquecimento tornaria a
        tabela de custo do relatorio incorreta, sem nenhum sinal de erro.
        Como callback, a contabilidade acontece por construcao.

    O callback le o consumo REAL informado pela API na resposta. Quando o
    provedor nao informa (caso do Ollama), estimamos pelo texto.
    """

    def __init__(self, modelo: str, origem: str = "nao_informada", provedor: str = "") -> None:
        self.modelo = modelo
        self.origem = origem
        self.provedor = provedor

    def on_llm_end(self, response, **kwargs: Any) -> None:  # noqa: ANN001
        uso: dict[str, Any] = {}

        # Cada versao de integracao coloca o uso em um lugar diferente.
        # Procuramos nos tres locais conhecidos antes de desistir.
        saida = getattr(response, "llm_output", None) or {}
        uso = saida.get("token_usage") or saida.get("usage") or {}

        if not uso:
            for geracoes in getattr(response, "generations", []):
                for geracao in geracoes:
                    metadados = getattr(geracao.message, "usage_metadata", None) if hasattr(geracao, "message") else None
                    if metadados:
                        uso = {
                            "prompt_tokens": metadados.get("input_tokens", 0),
                            "completion_tokens": metadados.get("output_tokens", 0),
                        }
                        break

        entrada = int(uso.get("prompt_tokens") or uso.get("input_tokens") or 0)
        saida_tokens = int(uso.get("completion_tokens") or uso.get("output_tokens") or 0)

        if not entrada and not saida_tokens:
            # Sem informacao do provedor: estimamos pelo texto gerado.
            texto = "".join(
                g.text for geracoes in getattr(response, "generations", []) for g in geracoes
            )
            saida_tokens = estimar_tokens(texto, self.modelo)

        contador().registrar_uso(
            self.modelo, entrada, saida_tokens, origem=self.origem, provedor=self.provedor
        )


# =============================================================================
# PROVEDOR ECO - modelo deterministico sem rede
# =============================================================================
class ChatEco(BaseChatModel):
    """
    Modelo falso que responde por heuristica, respeitando o formato de saida.

    NAO E UM MOCK DE TESTE COMUM. Um mock devolveria uma string fixa e os
    guardrails passariam trivialmente. Este devolve uma resposta com a mesma
    ESTRUTURA que o modelo real produz - decisao na primeira linha quando a
    tarefa pede, citacao das fontes presentes no contexto, aviso de validacao
    quando ha posologia. Assim o pipeline inteiro e exercitado de verdade,
    inclusive os caminhos de reprovacao.

    A escolha do rotulo e por palavra-chave no contexto. Nao tem valor
    clinico algum, e nem pretende ter: o proposito e a forma, nao o conteudo.
    """

    nome_modelo: str = "eco"

    @property
    def _llm_type(self) -> str:
        return "medgraph-eco"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        texto_usuario = "\n".join(
            str(m.content) for m in messages if m.type in ("human", "user")
        )
        resposta = self._responder(texto_usuario)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resposta))])

    @staticmethod
    def _responder(entrada: str) -> str:
        # Marcadores de fonte realmente presentes no contexto recebido.
        marcadores = re.findall(r"\[([EPC]\d+)\]", entrada)
        vistos: list[str] = []
        for m in marcadores:
            if m not in vistos:
                vistos.append(m)
        citacoes = ", ".join(f"[{m}]" for m in vistos) or "[E1]"

        baixo = entrada.lower()

        # Tarefa de decisao sobre evidencia: precisa da primeira linha fixa.
        if "decisão:" in baixo or "decisao:" in baixo:
            if any(t in baixo for t in ("no significant", "did not", "no difference", "failed to")):
                decisao = "no"
            elif any(t in baixo for t in ("inconclusive", "unclear", "further research", "limited evidence")):
                decisao = "maybe"
            else:
                decisao = "yes"
            return (
                f"Decisão: {decisao}\n"
                f"(modo eco, sem inferência real) A evidência fornecida foi considerada "
                f"para esta decisão.\n"
                f"Fontes: {citacoes}"
            )

        # Demais tarefas: resposta ancorada, com aviso quando houver posologia.
        tem_posologia = bool(re.search(r"\d+\s*(mg|mcg|g|ml|ui|mEq)\b", entrada, re.IGNORECASE))
        corpo = (
            "(modo eco, sem inferência real) Resposta baseada exclusivamente no "
            f"contexto fornecido {citacoes}."
        )
        if tem_posologia:
            corpo += (
                "\n\nEsta orientação envolve conduta terapêutica e depende de validação "
                "do médico responsável antes de qualquer prescrição."
            )
        return f"{corpo}\nFontes: {citacoes}"


# =============================================================================
# FABRICA
# =============================================================================
def _verificar_ollama(cfg: Settings) -> list[str]:
    """Confere que o servidor responde e devolve os modelos registrados."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{cfg.ollama_base_url}/api/tags", timeout=5) as resposta:
            return [m["name"] for m in json.loads(resposta.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProvedorIndisponivelError(
            f"O Ollama nao respondeu em {cfg.ollama_base_url} ({type(exc).__name__}).\n"
            f"Para resolver:\n"
            f"  brew services start ollama      # subir o servidor\n"
            f"  make modelo                     # registrar o modelo do projeto\n"
            f"Ou, para rodar sem o modelo local:\n"
            f"  LLM_PROVIDER=eco   no .env      # modo offline, sem inferencia\n"
            f"  LLM_PROVIDER=openai no .env     # usa gpt-4o-mini (custa centavos)"
        ) from exc


def obter_llm(
    cfg: Settings | None = None,
    *,
    origem: str = "nao_informada",
    temperatura: float | None = None,
    provedor: str | None = None,
) -> BaseChatModel:
    """
    Devolve o modelo de chat configurado.

    Args:
        origem: rotulo de onde a chamada nasceu, para a contabilidade de
            custo. Ex.: "grafo.raciocinio_clinico", "avaliacao.baseline".
        temperatura: sobrescreve a do .env. A avaliacao usa 0.0 para ser
            reproduzivel; o assistente usa o valor configurado.
        provedor: sobrescreve LLM_PROVIDER. Usado pela avaliacao, que compara
            provedores diferentes na mesma execucao.
    """
    cfg = cfg or obter_settings()
    escolhido = provedor or cfg.llm_provider
    temperatura = cfg.llm_temperature if temperatura is None else temperatura

    # -------------------------------------------------------------------------
    if escolhido == "eco":
        registrar(TipoEvento.LLM, "Provedor: eco (offline, sem inferencia)", origem=origem)
        return ChatEco()

    # -------------------------------------------------------------------------
    if escolhido == "ollama":
        from langchain_ollama import ChatOllama

        modelos = _verificar_ollama(cfg)
        if not any(m.split(":")[0] == cfg.ollama_model for m in modelos):
            raise ProvedorIndisponivelError(
                f"O modelo '{cfg.ollama_model}' nao esta registrado no Ollama.\n"
                f"Registrados: {modelos or '(nenhum)'}\n"
                f"Para registrar:  make modelo"
            )

        registrar(
            TipoEvento.LLM,
            f"Provedor: ollama ({cfg.ollama_model})",
            origem=origem,
            temperatura=temperatura,
        )
        return ChatOllama(
            model=cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=temperatura,
            num_predict=cfg.llm_max_tokens,
            callbacks=[CallbackCusto(cfg.ollama_model, origem, provedor="ollama")],
        )

    # -------------------------------------------------------------------------
    if escolhido == "openai":
        if not cfg.openai_configurada:
            raise ProvedorIndisponivelError(
                "LLM_PROVIDER=openai, mas OPENAI_API_KEY esta vazia no .env.\n"
                "Crie a chave em platform.openai.com dentro de um projeto com limite "
                "de gasto definido, e cole no .env."
            )

        from langchain_openai import ChatOpenAI

        # Consulta o orcamento ANTES de construir o cliente. Melhor recusar
        # aqui do que descobrir o estouro no meio de um laco de avaliacao.
        contador(cfg).verificar_orcamento()

        registrar(
            TipoEvento.LLM,
            f"Provedor: openai ({cfg.openai_model})",
            origem=origem,
            temperatura=temperatura,
            saldo_usd=round(contador(cfg).saldo_usd, 4),
        )
        return ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            temperature=temperatura,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout_s,
            callbacks=[CallbackCusto(cfg.openai_model, origem, provedor="openai")],
        )

    raise ValueError(
        f"Provedor desconhecido: {escolhido!r}. Use 'ollama', 'openai' ou 'eco'."
    )


def obter_llm_com_fallback(
    cfg: Settings | None = None,
    *,
    origem: str = "nao_informada",
    ordem: Sequence[str] = ("ollama", "openai", "eco"),
) -> tuple[BaseChatModel, str]:
    """
    Tenta os provedores em ordem e usa o primeiro disponivel.

    POR QUE EXISTE:
        Na apresentacao do trabalho, "o Ollama nao estava rodando" nao pode
        derrubar a demonstracao inteira. Com o encadeamento, o sistema degrada
        de forma visivel e controlada: o painel mostra qual provedor esta
        ativo, e o modo eco garante que o fluxo do grafo continue demonstravel
        mesmo sem nenhuma inferencia disponivel.

    Returns:
        O modelo e o nome do provedor efetivamente usado.
    """
    cfg = cfg or obter_settings()
    tentativas: list[str] = []

    # O provedor configurado tem prioridade sobre a ordem padrao.
    sequencia = [cfg.llm_provider] + [p for p in ordem if p != cfg.llm_provider]

    for candidato in sequencia:
        try:
            return obter_llm(cfg, origem=origem, provedor=candidato), candidato
        except (ProvedorIndisponivelError, ImportError, ValueError) as exc:
            tentativas.append(f"{candidato}: {type(exc).__name__}")
            log.warning("Provedor '%s' indisponivel, tentando o proximo.", candidato)

    raise ProvedorIndisponivelError(
        "Nenhum provedor de LLM disponivel. Tentativas: " + " | ".join(tentativas)
    )
