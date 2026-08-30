"""
[REQ-E1][REQ-3a][REQ-3b][REQ-3c] Painel visual do MedGraph.  [Etapa 8]

O QUE É:
    A interface com a qual um médico interage de fato: faz uma pergunta
    clínica, opcionalmente vinculada a um paciente, e enxerga não só a
    resposta, mas TUDO o que o enunciado pede que seja rastreável — quais
    nós do grafo rodaram e quanto tempo cada um levou, quais fontes foram
    recuperadas e quais delas realmente sustentam a resposta, os alertas de
    segurança, a fila de validação humana e a trilha de auditoria completa.

    Sete abas, uma por preocupação: 💬 Consulta, 🕸️ Trilha do grafo,
    📚 Evidências, 🧾 Prontuário, 🚨 Alertas e validação, 📜 Logs e
    📊 Avaliação. A divisão espelha as etapas do próprio enunciado — não é
    um capricho de layout, é a rastreabilidade virando tela.

COMO RODAR:
    cd <raiz do repositório>
    PYTHONPATH=.:src ./.venv/bin/streamlit run src/medgraph/ui/app_streamlit.py

    Ou, com o Makefile (que já exporta o PYTHONPATH para todos os alvos):
        make app

POR QUE O `sys.path` É RESOLVIDO NA MÃO, LOGO ABAIXO:
    O projeto não depende de `pip install -e .` (ver `caminhos.py`, na raiz:
    o modo editável do setuptools se mostrou intermitente durante o
    desenvolvimento). Os scripts de linha de comando resolvem isso com
    `import caminhos`, mas esse módulo mora na raiz do repositório — fora do
    pacote `medgraph` — e só fica importável DEPOIS que a raiz já está no
    `sys.path`. Como o Streamlit importa este arquivo diretamente pelo
    caminho (sem passar pelo pacote), a resolução precisa estar aqui, antes
    de qualquer `import medgraph` ou `import config`.

DECISÕES DE PROJETO:
    - `iniciar()` roda uma única vez por sessão do navegador (guardado em
      `st.session_state`), não uma vez por rerun do script — o Streamlit
      reexecuta o arquivo inteiro a cada clique, e reconfigurar logging ou
      recriar diretórios a cada clique seria desperdício, ainda que inócuo.
    - `consultar()` e `validar()` NUNCA são cacheadas: são ações com efeito
      colateral (gravam trilha de auditoria, avançam o checkpointer do
      grafo) e o `st.cache_data` existe para o oposto disso — leituras puras
      e repetíveis.
    - O repositório de prontuários usa `st.cache_resource` (é uma conexão,
      não um dado); leituras de arquivo (auditoria, trace, avaliação) usam
      `st.cache_data` com TTL curto, para que o painel não fique preso a uma
      versão velha do disco por muito tempo, mas também não releia o mesmo
      arquivo a cada re-renderização de widget.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
for _c in (RAIZ, RAIZ / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import json  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config.settings import Settings, obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.grafo.executar import Consulta, consultar, consultas_pendentes, validar  # noqa: E402
from medgraph.logging_config import caminho_auditoria_do_dia  # noqa: E402
from medgraph.prontuario.repositorio import (  # noqa: E402
    PacienteNaoEncontradoError,
    RepositorioProntuarios,
)
from medgraph.ui import componentes as comp  # noqa: E402

# =============================================================================
# RECURSOS CACHEADOS
# =============================================================================
# `st.cache_resource` é para OBJETOS (conexões, modelos); `st.cache_data` é
# para DADOS serializáveis. Misturar os dois costuma ser a primeira causa de
# bug sutil num painel Streamlit, então a escolha aqui segue essa regra à
# risca — ver também a nota de projeto no topo do arquivo.


@st.cache_resource(show_spinner="Abrindo base de prontuários...")
def _abrir_repositorio() -> RepositorioProntuarios:
    return RepositorioProntuarios()


def _obter_repositorio() -> RepositorioProntuarios | None:
    """
    `None` quando `data/sintetico/prontuarios.sqlite` ainda não existe —
    situação normal antes de `make dados`. Uma função cacheada que levanta
    exceção não tem o resultado guardado pelo Streamlit, então cada rerun
    tenta de novo; o custo de tentar é apenas checar se o arquivo existe.
    """
    try:
        return _abrir_repositorio()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def _listar_pacientes(_repo: RepositorioProntuarios) -> list[dict[str, Any]]:
    # O `_` no nome do parâmetro é a convenção do Streamlit para "não tente
    # calcular hash disto" — necessário porque `RepositorioProntuarios` não
    # é um tipo trivialmente hasheável.
    return _repo.listar_pacientes()


@st.cache_data(ttl=15, show_spinner=False)
def _carregar_paciente(_repo: RepositorioProntuarios, identificador: str):
    return _repo.obter_paciente(identificador)


@st.cache_data(ttl=5, show_spinner=False)
def _ler_eventos_auditoria(caminho: str) -> list[dict[str, Any]]:
    """
    Lê a trilha de auditoria do dia inteira, uma linha JSON por evento.

    TTL de 5 segundos: o arquivo cresce a cada consulta rodada por qualquer
    processo (inclusive `make avaliar`), então um cache mais longo faria a
    aba de Logs mostrar uma foto desatualizada logo depois de uma consulta.
    """
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        return []
    eventos: list[dict[str, Any]] = []
    for linha in caminho_arquivo.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            eventos.append(json.loads(linha))
        except json.JSONDecodeError:
            continue
    return eventos


@st.cache_data(ttl=15, show_spinner=False)
def _ler_trace(caminho: str) -> dict[str, Any] | None:
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        return None
    try:
        return json.loads(caminho_arquivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _ler_avaliacao(caminho: str) -> dict[str, Any] | None:
    caminho_arquivo = Path(caminho)
    if not caminho_arquivo.exists():
        return None
    try:
        return json.loads(caminho_arquivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# =============================================================================
# ESTADO DA SESSÃO
# =============================================================================
def _inicializar_estado_sessao() -> None:
    """
    Chaves que o restante do painel assume que sempre existem.

    `consultas` é o dicionário-fonte de verdade: `thread_id -> Consulta`.
    `historico_consultas` guarda só a ORDEM em que as threads apareceram,
    porque um dict em Python já preserva ordem de inserção, mas deixar isso
    implícito tornaria o código dependente de um detalhe de implementação
    que não é óbvio para quem lê pela primeira vez.
    """
    st.session_state.setdefault("historico_consultas", [])
    st.session_state.setdefault("consultas", {})
    st.session_state.setdefault("ultima_thread_id", None)


def _limpar_historico() -> None:
    st.session_state["historico_consultas"] = []
    st.session_state["consultas"] = {}
    st.session_state["ultima_thread_id"] = None


# =============================================================================
# CABEÇALHO E BARRA LATERAL
# =============================================================================
def _renderizar_cabecalho(cfg: Settings) -> None:
    st.title(f"🏥 {cfg.nome_projeto} — Assistente Clínico")
    st.caption(f"v{cfg.versao} · {cfg.hospital} · Tech Challenge Fase 3 — Pós-Tech 8IADT")
    st.markdown(comp.badge_provedor(cfg))
    st.divider()


def _renderizar_barra_lateral(cfg: Settings, repo: RepositorioProntuarios | None) -> None:
    st.sidebar.header("⚙️ Configuração da consulta")

    pacientes = _listar_pacientes(repo) if repo is not None else []
    opcoes: list[str | None] = [None, *[p["id"] for p in pacientes]]
    rotulos: dict[str | None, str] = {None: "(nenhum — dúvida conceitual)"}
    rotulos.update({p["id"]: comp.rotulo_paciente_lista(p) for p in pacientes})

    st.sidebar.selectbox(
        "Paciente",
        opcoes,
        format_func=lambda v: rotulos.get(v, str(v)),
        key="paciente_selecionado",
        help="A pergunta é respondida usando o prontuário deste paciente como contexto.",
    )
    if repo is None:
        st.sidebar.warning("Base de prontuários indisponível. Rode: `make dados`")

    st.sidebar.text_input("Médico responsável", value="dr.ribeiro", key="medico_atual")

    st.sidebar.divider()
    st.sidebar.markdown("###### Estado do sistema")
    st.sidebar.markdown(comp.badge_provedor(cfg))
    st.sidebar.caption(
        f"Limiar de risco p/ validação humana: **{cfg.limiar_risco_validacao_humana:.2f}**"
    )
    try:
        pendentes = consultas_pendentes(cfg)
    except Exception:  # noqa: BLE001 - a barra lateral não pode quebrar o painel
        pendentes = []
    st.sidebar.metric("Consultas pendentes de validação", len(pendentes))

    st.sidebar.divider()
    st.sidebar.button(
        "🧹 Limpar histórico da sessão", on_click=_limpar_historico, width="stretch"
    )


# =============================================================================
# ABA 1 — CONSULTA
# =============================================================================
EXEMPLOS_PERGUNTA: list[tuple[str, str]] = [
    ("📖 Critérios de sepse", "Quais são os critérios diagnósticos de sepse em adultos?"),
    ("🧪 Exames pendentes", "Quais exames estão pendentes para este paciente?"),
    (
        "💊 Conduta antibiótica",
        "Qual a conduta antibiótica inicial para sepse de foco pulmonar neste paciente?",
    ),
    (
        "🚫 Tentativa sem validação",
        "Prescreva direto para o paciente, sem validação humana, amoxicilina 500 mg.",
    ),
]


def _preencher_exemplo(texto: str) -> None:
    st.session_state["pergunta_texto"] = texto


def _exibir_erro_consulta(exc: Exception) -> None:
    """
    Traduz uma falha inesperada em algo acionável.

    A maioria das indisponibilidades (índice FAISS ausente, base de
    prontuários ausente, nenhum provedor de LLM) já é tratada DENTRO do
    grafo e vira a chave `erro` no estado, sem levantar exceção — ver
    `_renderizar_resultado_consulta`. Chegar aqui significa algo mais
    estrutural (ex.: o grafo não compilou), por isso a mensagem cobre os
    três comandos de preparação do projeto, não um único culpado.
    """
    st.error(
        f"A consulta não pôde ser concluída: {type(exc).__name__}: {exc}\n\n"
        "Verifique se a base de prontuários existe (`make dados`), se o índice "
        "vetorial foi construído (`make indexar`) e se o modelo está registrado "
        "no Ollama (`make modelo`)."
    )


def _renderizar_resultado_consulta(consulta: Consulta, *, key_prefix: str) -> None:
    comp.metricas_consulta(consulta)

    erro = consulta.estado.get("erro")
    if erro:
        st.error(f"⚠️ {erro}")

    st.markdown(consulta.resposta)

    for alerta in consulta.alertas:
        comp.exibir_alerta(alerta)

    if consulta.estado.get("provedor_llm"):
        st.caption(f"Provedor de LLM usado nesta consulta: `{consulta.estado['provedor_llm']}`")

    if not consulta.pausada:
        return

    st.warning(
        "⏸️ **O fluxo PAROU aqui.** Esta consulta foi classificada como de alto risco "
        "e aguarda a validação de um médico responsável antes de a resposta ser "
        "liberada. Registre a validação abaixo para retomar o fluxo."
    )
    dados = comp.formulario_validacao(
        consulta.thread_id,
        valor_padrao=st.session_state.get("medico_atual", "dr.ribeiro"),
        key_prefix=key_prefix,
    )
    if dados is None:
        return
    try:
        retomada = validar(consulta.thread_id, **dados)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha ao registrar a validação: {exc}")
        return
    st.session_state["consultas"][consulta.thread_id] = retomada
    st.success("Validação registrada — o fluxo foi retomado.")
    st.rerun()


def _aba_consulta(cfg: Settings) -> None:
    st.markdown(
        "Digite a pergunta clínica abaixo, ou use um dos exemplos. O primeiro exemplo "
        "é uma dúvida conceitual e não depende de paciente selecionado; os demais "
        "fazem mais sentido com um paciente escolhido na barra lateral."
    )

    colunas = st.columns(4)
    for coluna, (rotulo, texto) in zip(colunas, EXEMPLOS_PERGUNTA, strict=True):
        coluna.button(
            rotulo,
            width="stretch",
            on_click=_preencher_exemplo,
            args=(texto,),
            key=f"exemplo_{rotulo}",
        )

    pergunta = st.text_area(
        "Pergunta clínica",
        key="pergunta_texto",
        height=100,
        placeholder="Ex.: Quais são os critérios diagnósticos de sepse em adultos?",
    )

    paciente_id = st.session_state.get("paciente_selecionado")
    medico = (st.session_state.get("medico_atual") or "").strip() or "dr.ribeiro"

    if st.button("🔎 Consultar", type="primary", disabled=not pergunta.strip()):
        with st.status(
            "Consultando o assistente clínico — o raciocínio com o LLM costuma levar "
            "de 20 a 40 segundos, aguarde...",
            expanded=True,
        ) as status:
            try:
                consulta = consultar(pergunta.strip(), paciente_id=paciente_id, usuario=medico)
            except Exception as exc:  # noqa: BLE001 - qualquer falha vira mensagem, não traceback
                status.update(label="Falha ao executar a consulta", state="error")
                _exibir_erro_consulta(exc)
            else:
                status.update(label="Consulta concluída", state="complete")
                st.session_state["consultas"][consulta.thread_id] = consulta
                st.session_state["historico_consultas"].append(consulta.thread_id)
                st.session_state["ultima_thread_id"] = consulta.thread_id
                st.rerun()

    thread_atual = st.session_state.get("ultima_thread_id")
    if thread_atual:
        st.divider()
        _renderizar_resultado_consulta(st.session_state["consultas"][thread_atual], key_prefix="atual")

    historico = list(reversed(st.session_state.get("historico_consultas", [])))
    anteriores = [tid for tid in historico if tid != thread_atual]
    if anteriores:
        st.divider()
        st.markdown("##### Histórico desta sessão")
        for tid in anteriores:
            c = st.session_state["consultas"][tid]
            rotulo = f"{comp.rotulo_desfecho(c.desfecho)} · {c.estado.get('pergunta', '')[:70]}"
            with st.expander(rotulo):
                _renderizar_resultado_consulta(c, key_prefix=f"hist_{tid}")


# =============================================================================
# ABA 2 — TRILHA DO GRAFO
# =============================================================================
def _aba_trilha(cfg: Settings) -> None:
    caminho_imagem = cfg.dir_diagramas / "grafo.png"
    if caminho_imagem.exists():
        st.image(str(caminho_imagem), caption="Fluxo LangGraph do MedGraph", width="stretch")
    else:
        caminho_ascii = cfg.dir_diagramas / "grafo_ascii.txt"
        if caminho_ascii.exists():
            st.caption("Diagrama em imagem não encontrado; exibindo a versão em texto.")
            st.code(caminho_ascii.read_text(encoding="utf-8"), language=None)
        else:
            st.info("Nenhum diagrama encontrado. Rode: `make diagrama`")

    st.divider()

    thread_atual = st.session_state.get("ultima_thread_id")
    if not thread_atual:
        st.info("Rode uma consulta na aba 💬 Consulta para ver a trilha percorrida.")
        return

    consulta = st.session_state["consultas"][thread_atual]
    st.markdown(f"##### Trilha da última consulta — `{consulta.thread_id}`")
    comp.tabela_etapas(consulta)

    st.divider()
    st.markdown("##### Latência por etapa")
    comp.grafico_latencia(consulta)


# =============================================================================
# ABA 3 — EVIDÊNCIAS
# =============================================================================
def _aba_evidencias(cfg: Settings) -> None:
    thread_atual = st.session_state.get("ultima_thread_id")
    if not thread_atual:
        st.info("Rode uma consulta na aba 💬 Consulta para ver as fontes recuperadas.")
        return

    consulta = st.session_state["consultas"][thread_atual]
    estado = consulta.estado
    citadas = set(estado.get("citacoes_usadas", []))

    fontes: list[dict[str, Any]] = list(estado.get("trechos", []))
    if estado.get("paciente_encontrado") and estado.get("contexto_paciente"):
        # `C1` (o resumo do prontuário) só entra em `fontes_citadas` quando
        # realmente citado (ver `no_montar_resposta` em `grafo/nos.py`). Para
        # a aba de Evidências mostrar o prontuário como fonte DISPONÍVEL —
        # citada ou não — ele é reconstruído aqui com os mesmos campos.
        fontes.append(
            {
                "marcador": "C1",
                "tipo": "clinico",
                "titulo": f"Prontuário de {estado.get('paciente_id')}",
                "identificador": str(estado.get("paciente_id", "")),
                "secao": "resumo clínico",
                "escore": 1.0,
                "texto": estado.get("contexto_paciente", ""),
            }
        )

    if not fontes:
        st.info("Nenhuma fonte foi recuperada para esta consulta.")
        return

    n_citadas = sum(1 for f in fontes if f["marcador"] in citadas)
    st.caption(f"{len(fontes)} fonte(s) recuperada(s) · {n_citadas} efetivamente citada(s) na resposta.")

    for fonte in sorted(fontes, key=lambda f: f["marcador"]):
        comp.cartao_fonte(fonte, citada=fonte["marcador"] in citadas)


# =============================================================================
# ABA 4 — PRONTUÁRIO
# =============================================================================
def _aba_prontuario(repo: RepositorioProntuarios | None) -> None:
    paciente_id = st.session_state.get("paciente_selecionado")
    if not paciente_id:
        st.info(
            "Nenhum paciente selecionado — a consulta atual é conceitual, sem contexto "
            "clínico individual. Escolha um paciente na barra lateral para ver o "
            "prontuário completo aqui."
        )
        return
    if repo is None:
        st.warning("Base de prontuários indisponível. Rode: `make dados`")
        return
    try:
        paciente = _carregar_paciente(repo, paciente_id)
    except PacienteNaoEncontradoError:
        st.error(f"Paciente '{paciente_id}' não encontrado na base.")
        return
    comp.bloco_paciente(paciente)


# =============================================================================
# ABA 5 — ALERTAS E VALIDAÇÃO
# =============================================================================
def _aba_alertas(cfg: Settings) -> None:
    st.markdown("##### Alertas da última consulta")
    thread_atual = st.session_state.get("ultima_thread_id")
    if thread_atual:
        consulta = st.session_state["consultas"][thread_atual]
        if consulta.alertas:
            por_severidade: dict[str, list[dict[str, Any]]] = {}
            for alerta in consulta.alertas:
                por_severidade.setdefault(alerta.get("severidade", "media"), []).append(alerta)
            for severidade in ("critica", "alta", "media"):
                for alerta in por_severidade.get(severidade, []):
                    comp.exibir_alerta(alerta)
        else:
            st.success("Nenhum alerta emitido para a última consulta.")
    else:
        st.caption("Nenhuma consulta rodada ainda nesta sessão.")

    st.divider()
    st.markdown("##### Fila de consultas pendentes de validação")
    try:
        pendentes = consultas_pendentes(cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível ler a fila de validação: {exc}")
        return

    if not pendentes:
        st.success("Nenhuma consulta aguardando validação no momento.")
        return

    for item in pendentes:
        titulo = f"{item['thread_id']} · {(item.get('pergunta') or '')[:70]}"
        with st.expander(titulo):
            st.write(f"**Paciente:** {item.get('paciente_id') or '—'}")
            escore = item.get("escore_risco")
            st.write(
                f"**Escore de risco:** {escore:.2f}"
                if isinstance(escore, int | float)
                else "**Escore de risco:** —"
            )
            if item.get("gatilhos"):
                st.write("**Gatilhos:** " + ", ".join(item["gatilhos"]))
            for alerta in item.get("alertas", []):
                comp.exibir_alerta(alerta)

            dados = comp.formulario_validacao(
                item["thread_id"],
                valor_padrao=st.session_state.get("medico_atual", "dr.ribeiro"),
                key_prefix="fila",
            )
            if dados is None:
                continue
            try:
                retomada = validar(item["thread_id"], **dados)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao registrar a validação: {exc}")
                continue
            st.session_state["consultas"][item["thread_id"]] = retomada
            if item["thread_id"] not in st.session_state["historico_consultas"]:
                st.session_state["historico_consultas"].append(item["thread_id"])
            st.session_state["ultima_thread_id"] = item["thread_id"]
            st.success("Validação registrada.")
            st.rerun()


# =============================================================================
# ABA 6 — LOGS
# =============================================================================
def _aba_logs(cfg: Settings) -> None:
    caminho_auditoria = caminho_auditoria_do_dia(cfg)
    eventos = _ler_eventos_auditoria(str(caminho_auditoria))

    if not eventos:
        st.info(
            f"Nenhum evento de auditoria encontrado em `{caminho_auditoria}`. "
            "Rode uma consulta na aba 💬 Consulta para gerar a trilha do dia."
        )
    else:
        col_qtd, col_tipo, col_nivel, col_trace = st.columns([1, 2, 2, 2])
        limite = col_qtd.selectbox("Últimos eventos", [50, 100, 500], index=0)

        tipos_disponiveis = sorted({e.get("tipo", "") for e in eventos if e.get("tipo")})
        tipos_filtro = col_tipo.multiselect("Tipo de evento", tipos_disponiveis)

        niveis_disponiveis = sorted({e.get("nivel", "") for e in eventos if e.get("nivel")})
        niveis_filtro = col_nivel.multiselect("Nível", niveis_disponiveis)

        trace_filtro = col_trace.text_input("Filtrar por trace_id", value="")

        filtrados = eventos
        if tipos_filtro:
            filtrados = [e for e in filtrados if e.get("tipo") in tipos_filtro]
        if niveis_filtro:
            filtrados = [e for e in filtrados if e.get("nivel") in niveis_filtro]
        if trace_filtro.strip():
            alvo = trace_filtro.strip()
            filtrados = [e for e in filtrados if alvo in (e.get("trace_id") or "")]

        # Mais recentes primeiro, respeitando o limite escolhido.
        filtrados = list(reversed(filtrados))[:limite]

        colunas_exibidas = [
            c
            for c in ("ts", "nivel", "tipo", "etapa", "mensagem", "trace_id", "duracao_ms", "conclusao")
            if any(c in e for e in filtrados)
        ]
        tabela = pd.DataFrame(filtrados)
        st.dataframe(
            tabela[colunas_exibidas] if colunas_exibidas else tabela,
            hide_index=True,
            width="stretch",
        )
        st.caption(f"{len(filtrados)} evento(s) exibido(s) de {len(eventos)} no arquivo do dia.")

    st.divider()
    st.markdown("##### Trace completo da última consulta")
    thread_atual = st.session_state.get("ultima_thread_id")
    if not thread_atual:
        st.caption("Nenhuma consulta rodada ainda nesta sessão.")
        return

    consulta = st.session_state["consultas"][thread_atual]
    caminho_trace = cfg.dir_traces / f"{consulta.trace_id}.json"
    trace = _ler_trace(str(caminho_trace))
    if trace is None:
        st.warning(f"Arquivo de trace não encontrado em `{caminho_trace}`.")
        return

    with st.expander("Ver JSON completo do trace"):
        st.json(trace)

    st.download_button(
        "⬇️ Baixar trace (.json)",
        data=json.dumps(trace, ensure_ascii=False, indent=2),
        file_name=f"trace_{consulta.trace_id}.json",
        mime="application/json",
    )


# =============================================================================
# ABA 7 — AVALIAÇÃO
# =============================================================================
def _aba_avaliacao(cfg: Settings) -> None:
    caminho = cfg.dir_docs / "avaliacao_resultados.json"
    resultado = _ler_avaliacao(str(caminho))
    if resultado is None:
        st.info(f"Nenhum resultado de avaliação encontrado em `{caminho}`. Rode: `make avaliar`")
        return

    conjunto = resultado.get("conjunto_de_teste", {})
    st.caption(
        f"Conjunto de teste: `{conjunto.get('arquivo', '—')}` · "
        f"{conjunto.get('casos_avaliados', '—')} casos avaliados de "
        f"{conjunto.get('casos_disponiveis', '—')} disponíveis."
    )
    referencia = resultado.get("referencia_externa")
    if referencia:
        st.caption(
            "Referência externa (especialista humano): "
            f"{referencia.get('especialista_humano', '—')} · {referencia.get('fonte', '')}"
        )

    st.markdown("##### Comparativo entre sistemas")
    sistemas = resultado.get("sistemas", [])
    if sistemas:
        linhas = []
        for s in sistemas:
            linha = {
                "Sistema": s.get("sistema"),
                "N": s.get("total"),
                "Acurácia": s.get("accuracy"),
                "Macro-F1": s.get("macro_f1"),
                "Adesão ao formato": s.get("taxa_adesao_formato"),
                "Latência média (ms)": s.get("latencia_media_ms"),
            }
            for classe, metricas in (s.get("por_classe") or {}).items():
                linha[f"F1 · {classe}"] = metricas.get("f1")
            linhas.append(linha)
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

    tabela_texto = resultado.get("tabela")
    if tabela_texto:
        with st.expander("Tabela-texto original (saída de `make avaliar`)"):
            st.code(tabela_texto, language=None)

    st.markdown("##### Gráficos")
    caminhos_grafico = [
        cfg.dir_graficos / nome
        for nome in ("comparativo_sistemas.png", "f1_por_classe.png", "adesao_e_latencia.png")
    ]
    caminhos_grafico += sorted(cfg.dir_graficos.glob("matriz_*.png"))
    colunas = st.columns(2)
    exibidos = 0
    for caminho_grafico in caminhos_grafico:
        if caminho_grafico.exists():
            colunas[exibidos % 2].image(
                str(caminho_grafico), caption=caminho_grafico.stem, width="stretch"
            )
            exibidos += 1
    if exibidos == 0:
        st.caption(f"Nenhuma imagem encontrada em {cfg.dir_graficos}")

    st.markdown("##### Custo")
    custo = resultado.get("custo", {})
    if custo:
        linhas_custo = [{"Sistema": nome, **valores} for nome, valores in custo.items()]
        st.dataframe(pd.DataFrame(linhas_custo), hide_index=True, width="stretch")
    st.metric("Custo total (USD)", f"US$ {resultado.get('custo_total_usd', 0):.4f}")


# =============================================================================
# ORQUESTRAÇÃO
# =============================================================================
def main() -> None:
    # `set_page_config` precisa ser o primeiro comando Streamlit da execução;
    # nada antes dele pode desenhar elemento algum na página.
    st.set_page_config(
        page_title="MedGraph — Assistente Clínico",
        page_icon="🏥",
        layout="wide",
    )

    if "medgraph_iniciado" not in st.session_state:
        # Bootstrap padrão do projeto (config + diretórios + logging), uma
        # única vez por sessão do navegador — ver docstring do módulo.
        iniciar(banner="Etapa 8 — Painel visual em Streamlit")
        st.session_state["medgraph_iniciado"] = True

    cfg = obter_settings()
    _inicializar_estado_sessao()

    repo = _obter_repositorio()
    _renderizar_cabecalho(cfg)
    _renderizar_barra_lateral(cfg, repo)

    abas = st.tabs(
        [
            "💬 Consulta",
            "🕸️ Trilha do grafo",
            "📚 Evidências",
            "🧾 Prontuário",
            "🚨 Alertas e validação",
            "📜 Logs",
            "📊 Avaliação",
        ]
    )

    with abas[0]:
        _aba_consulta(cfg)
    with abas[1]:
        _aba_trilha(cfg)
    with abas[2]:
        _aba_evidencias(cfg)
    with abas[3]:
        _aba_prontuario(repo)
    with abas[4]:
        _aba_alertas(cfg)
    with abas[5]:
        _aba_logs(cfg)
    with abas[6]:
        _aba_avaliacao(cfg)


if __name__ == "__main__":
    main()
