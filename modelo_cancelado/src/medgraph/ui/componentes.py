"""
[REQ-E1][REQ-3c] Componentes visuais reutilizados pelo painel Streamlit.

O QUE FAZ:
    Reúne as peças de interface que aparecem em mais de uma aba — cartão de
    fonte recuperada, bloco completo do prontuário, tabela de etapas do
    grafo, gráfico de latência e o formulário de validação médica.

POR QUE SEPARADO DE `app_streamlit.py`:
    O arquivo principal já orquestra sete abas; deixar a MONTAGEM de cada
    tela junto com a RENDERIZAÇÃO de cada peça visual tornaria os dois
    difíceis de ler separadamente. Aqui ficam funções pequenas e sem estado
    (recebem os dados prontos e apenas desenham), o que também as torna mais
    fáceis de testar isoladamente se um dia isso for necessário.

    Nenhuma função deste módulo chama `consultar()`, `validar()` ou qualquer
    outra função com efeito colateral sobre o grafo — só leem dados e
    desenham. Quem decide O QUE fazer com uma submissão de formulário é
    sempre o `app_streamlit.py`, que tem acesso ao `st.session_state`.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import altair as alt
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    # Import só para checagem de tipos: evita ciclo de import em tempo de
    # execução, já que estes módulos não precisam deste aqui.
    from config.settings import Settings
    from medgraph.grafo.executar import Consulta
    from medgraph.prontuario.modelos import Paciente

# =============================================================================
# VOCABULÁRIO VISUAL
# =============================================================================
# Centralizado aqui para que um ícone ou rótulo não seja escrito de um jeito
# na aba de Evidências e de outro na aba de Alertas.

ICONE_TIPO_FONTE: dict[str, str] = {"evidencia": "🔬", "protocolo": "📋", "clinico": "🧾"}
ROTULO_TIPO_FONTE: dict[str, str] = {
    "evidencia": "Evidência científica",
    "protocolo": "Protocolo interno",
    "clinico": "Prontuário do paciente",
}
ICONE_SEVERIDADE: dict[str, str] = {"critica": "🟥", "alta": "🟧", "media": "🟨"}
ROTULO_DESFECHO: dict[str, str] = {
    "respondida": "✅ Respondida",
    "recusada": "🚫 Recusada pelo guardrail",
    "aguardando_validacao": "⏸️ Aguardando validação",
    "degradada": "⚠️ Degradada",
    "erro": "❌ Erro",
}

# Acima deste tempo, uma etapa é destacada como lenta na trilha do grafo.
LIMIAR_LENTIDAO_MS = 5000.0


def formatar_ms(ms: float | None) -> str:
    """Converte milissegundos num texto curto, em segundos quando é o caso."""
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.0f} ms"


def rotulo_desfecho(desfecho: str) -> str:
    return ROTULO_DESFECHO.get(desfecho, desfecho)


def rotulo_paciente_lista(paciente: dict[str, Any]) -> str:
    """
    Rótulo do seletor de paciente da barra lateral.

    O ⚠️ sinaliza, sem precisar abrir o prontuário, que aquele paciente tem
    alergia registrada ou exame em valor crítico — exatamente os dois fatos
    que mais mudam a conduta de uma resposta clínica.
    """
    alerta = " ⚠️" if paciente.get("qtd_exames_criticos") or paciente.get("qtd_alergias") else ""
    return (
        f"{paciente['id']} · {paciente['idade']}a · {paciente['setor']} · "
        f"{paciente['qtd_alergias']} alergia(s){alerta}"
    )


def badge_provedor(cfg: Settings) -> str:
    """Selo compacto do provedor de LLM configurado, para o cabeçalho."""
    if cfg.llm_provider == "ollama":
        modelo = cfg.ollama_model
    elif cfg.llm_provider == "openai":
        modelo = cfg.openai_model
    else:
        modelo = "heurística offline, sem LLM"
    return f"🧠 **{cfg.llm_provider}** · `{modelo}`"


# =============================================================================
# ALERTAS
# =============================================================================
def exibir_alerta(alerta: dict[str, Any]) -> None:
    """
    Desenha um alerta clínico com a caixa do Streamlit correspondente à
    severidade — crítica/alta viram `st.error`, o resto vira `st.warning`.

    O projeto só produz as severidades "critica", "alta" e "media" (ver
    `nos.py`), mas o `.get` com default cobre um valor inesperado sem
    quebrar o painel.
    """
    severidade = alerta.get("severidade", "media")
    icone = ICONE_SEVERIDADE.get(severidade, "⬜")
    corpo = f"{icone} **[{severidade.upper()}] {alerta.get('titulo', '(sem título)')}**\n\n"
    corpo += alerta.get("detalhe", "")
    if alerta.get("acao"):
        corpo += f"\n\n**Conduta recomendada:** {alerta['acao']}"

    if severidade in ("critica", "alta"):
        st.error(corpo)
    else:
        st.warning(corpo)


# =============================================================================
# EVIDÊNCIAS
# =============================================================================
def cartao_fonte(fonte: dict[str, Any], *, citada: bool) -> None:
    """
    Um cartão por trecho recuperado: marcador, tipo, procedência e o texto.

    A barra de similaridade usa o escore já convertido para "maior é melhor"
    pelo recuperador (ver `medgraph/rag/recuperador.py`); aqui só limitamos a
    [0, 1] porque `st.progress` rejeita valores fora dessa faixa, e um
    escore levemente negativo é matematicamente possível na conversão de
    distância L2 para similaridade de cosseno.
    """
    marcador = fonte.get("marcador", "?")
    tipo = fonte.get("tipo", "")
    icone = ICONE_TIPO_FONTE.get(tipo, "📄")
    rotulo_tipo = ROTULO_TIPO_FONTE.get(tipo, tipo or "fonte")
    escore = max(0.0, min(1.0, float(fonte.get("escore") or 0.0)))

    with st.container(border=True):
        col_titulo, col_selo = st.columns([3, 1])
        with col_titulo:
            st.markdown(f"#### [{marcador}] {icone} {rotulo_tipo}")
            rodape = fonte.get("identificador", "—")
            if fonte.get("secao"):
                rodape += f" · {fonte['secao']}"
            st.caption(rodape)
        with col_selo:
            if citada:
                st.success("✅ citada na resposta")
            else:
                st.info("◻️ recuperada, não citada")

        st.progress(escore, text=f"Similaridade: {escore:.2f}")
        st.markdown(fonte.get("texto", "").strip() or "_(trecho sem texto)_")


# =============================================================================
# TRILHA DO GRAFO
# =============================================================================
def tabela_etapas(consulta: Consulta) -> None:
    """
    Lista as etapas percorridas, na ordem, com uma barra proporcional à
    latência e destaque em vermelho para quem passou de 5 segundos.

    NÓS REPETIDOS (ciclo de reescrita):
        `Consulta.etapas` guarda a sequência completa, então um nó que
        rodou duas vezes aparece duas vezes aqui — é assim que o ciclo de
        reescrita fica visível. Já `Consulta.tempo_por_etapa` é um
        dicionário indexado pelo NOME da etapa (ver `TrilhaAuditoria.
        tempo_por_etapa` em `auditoria.py`): quando o mesmo nó roda de novo,
        a segunda passagem sobrescreve a duração da primeira. Por isso a
        latência mostrada para uma etapa repetida é sempre a da ÚLTIMA
        passagem — uma limitação do dado de origem, não deste painel.
    """
    etapas = consulta.etapas
    if not etapas:
        st.caption("Nenhuma etapa registrada para esta consulta.")
        return

    contagem = Counter(etapas)
    maior_latencia = max(consulta.tempo_por_etapa.values(), default=0.0) or 1.0
    ocorrencia: Counter[str] = Counter()

    for nome in etapas:
        ocorrencia[nome] += 1
        repetida = contagem[nome] > 1
        ms = consulta.tempo_por_etapa.get(nome)

        col_nome, col_barra, col_ms = st.columns([2.4, 4, 1])
        rotulo = f"`{nome}`"
        if repetida:
            rotulo = f"🔁 {rotulo} — passagem {ocorrencia[nome]}/{contagem[nome]}"
        col_nome.markdown(rotulo)
        col_barra.progress(min(1.0, (ms or 0.0) / maior_latencia))
        texto_ms = formatar_ms(ms)
        col_ms.markdown(f":red[**{texto_ms}**]" if (ms or 0) > LIMIAR_LENTIDAO_MS else texto_ms)

    repetidos = [nome for nome, vezes in contagem.items() if vezes > 1]
    if repetidos:
        st.info(
            "🔁 O ciclo de reescrita executou novamente: **" + ", ".join(repetidos) + "**. "
            "A latência acima reflete apenas a última passagem de cada nó repetido."
        )


def grafico_latencia(consulta: Consulta) -> None:
    """
    Barras horizontais com a latência de cada etapa (nomes únicos).

    Usa Altair (e não `st.bar_chart`) porque precisamos colorir cada barra
    individualmente conforme ela passa ou não do limiar de 5 s — algo que
    o atalho `st.bar_chart` não expõe sem truques. Altair já é dependência
    do próprio Streamlit, então isso não adiciona peso ao projeto.
    """
    if not consulta.tempo_por_etapa:
        st.caption("Sem dados de latência para montar o gráfico.")
        return

    dados = pd.DataFrame(
        [{"etapa": nome, "latencia_ms": ms} for nome, ms in consulta.tempo_por_etapa.items()]
    )
    dados["lenta"] = dados["latencia_ms"] > LIMIAR_LENTIDAO_MS

    grafico = (
        alt.Chart(dados)
        .mark_bar()
        .encode(
            x=alt.X("latencia_ms:Q", title="Latência (ms)"),
            y=alt.Y("etapa:N", sort="-x", title=""),
            color=alt.condition(
                alt.datum.lenta, alt.value("#d62728"), alt.value("#4c78a8")
            ),
            tooltip=[alt.Tooltip("etapa:N", title="Etapa"), alt.Tooltip("latencia_ms:Q", title="ms")],
        )
        .properties(height=max(120, 32 * len(dados)))
    )
    st.altair_chart(grafico, width="stretch")

    lentas = dados.loc[dados["lenta"], "etapa"].tolist()
    if lentas:
        st.caption("🔴 Acima de 5 s: " + ", ".join(lentas))


# =============================================================================
# PRONTUÁRIO
# =============================================================================
def bloco_paciente(paciente: Paciente) -> None:
    """
    Desenha o prontuário inteiro: identificação, população especial,
    comorbidades, alergias, medicações, exames e evoluções.

    A gravidade "grave" de uma alergia vira `st.error` propositalmente —
    é a única informação do prontuário capaz, sozinha, de contraindicar uma
    conduta inteira.
    """
    st.subheader(f"{paciente.nome} · {paciente.id} · {paciente.prontuario}")
    col_idade, col_sexo, col_setor, col_leito = st.columns(4)
    col_idade.metric("Idade", f"{paciente.idade} anos")
    col_sexo.metric("Sexo", paciente.sexo)
    col_setor.metric("Setor", paciente.setor)
    col_leito.metric("Leito", paciente.leito or "—")

    marcadores = []
    if paciente.gestante:
        marcadores.append("🤰 GESTANTE")
    if paciente.pediatrico:
        marcadores.append("🧒 PEDIÁTRICO")
    if paciente.idade >= 80:
        marcadores.append("👴 IDOSO ≥ 80 ANOS")
    if marcadores:
        st.warning("**População especial:** " + " · ".join(marcadores))

    st.divider()

    st.markdown("##### Comorbidades")
    if paciente.comorbidades:
        for c in paciente.comorbidades:
            linha = c.descricao
            if c.cid10:
                linha += f" ({c.cid10})"
            if c.desde:
                linha += f" — desde {c.desde}"
            st.markdown(f"- {linha}")
    else:
        st.caption("Nenhuma comorbidade registrada.")

    st.markdown("##### Alergias")
    if paciente.alergias:
        for a in paciente.alergias:
            texto = f"**{a.substancia}**"
            if a.classe:
                texto += f" [{a.classe}]"
            if a.reacao:
                texto += f" — {a.reacao}"
            if a.e_grave:
                st.error(f"⚠️ GRAVE · {texto}")
            else:
                sufixo = f" _({a.gravidade})_" if a.gravidade else ""
                st.markdown(f"- {texto}{sufixo}")
    else:
        st.caption("Nenhuma alergia registrada.")

    st.markdown("##### Medicações ativas")
    ativas = paciente.medicacoes_ativas
    if ativas:
        for m in ativas:
            st.markdown(f"- {m}")
    else:
        st.caption("Nenhuma medicação ativa.")

    st.markdown("##### Exames")
    criticos = paciente.exames_criticos
    alterados = [e for e in paciente.exames if e.fora_da_faixa and not e.critico]
    pendentes = paciente.exames_pendentes
    col_criticos, col_alterados, col_pendentes = st.columns(3)
    with col_criticos:
        st.markdown("**🔴 Críticos**")
        for e in criticos:
            st.error(str(e))
        if not criticos:
            st.caption("Nenhum.")
    with col_alterados:
        st.markdown("**🟡 Alterados**")
        for e in alterados:
            st.warning(str(e))
        if not alterados:
            st.caption("Nenhum.")
    with col_pendentes:
        st.markdown("**⏳ Pendentes**")
        for e in pendentes:
            st.info(str(e))
        if not pendentes:
            st.caption("Nenhum.")

    st.markdown("##### Sinais vitais")
    if paciente.sinais_vitais:
        linhas = [{"Aferido em": s.aferido_em, "Valores": str(s)} for s in paciente.sinais_vitais]
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
    else:
        st.caption("Nenhum sinal vital registrado.")

    st.markdown("##### Evoluções")
    if paciente.evolucoes:
        for ev in paciente.evolucoes:
            titulo = f"{ev.data} — {ev.autor or 'autor não informado'} ({ev.especialidade or '—'})"
            with st.expander(titulo):
                st.write(ev.texto)
    else:
        st.caption("Nenhuma evolução registrada.")


# =============================================================================
# CONSULTA — MÉTRICAS E FORMULÁRIO DE VALIDAÇÃO
# =============================================================================
def metricas_consulta(consulta: Consulta) -> None:
    """Linha de métricas no topo do resultado de uma consulta."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Desfecho", rotulo_desfecho(consulta.desfecho))
    col2.metric("Duração", formatar_ms(consulta.duracao_ms))
    col3.metric("Etapas percorridas", str(len(consulta.etapas)))
    escore = consulta.estado.get("escore_risco")
    col4.metric("Escore de risco", f"{escore:.2f}" if isinstance(escore, int | float) else "—")
    col5.metric("Reescritas", str(consulta.estado.get("tentativas_reescrita", 0)))


def formulario_validacao(
    thread_id: str, *, valor_padrao: str, key_prefix: str
) -> dict[str, str] | None:
    """
    Formulário de validação médica: nome do validador + parecer.

    Devolve o dicionário submetido (ou `None` enquanto não houve submissão
    válida). Só desenha o formulário — quem chama `validar()` com o
    resultado é o `app_streamlit.py`, que também decide o que fazer com o
    `thread_id` no `st.session_state`.
    """
    with st.form(key=f"form_validacao_{key_prefix}_{thread_id}"):
        validado_por = st.text_input(
            "Médico responsável pela validação",
            value=valor_padrao,
            key=f"validador_{key_prefix}_{thread_id}",
        )
        parecer = st.text_area(
            "Parecer (opcional)", key=f"parecer_{key_prefix}_{thread_id}", height=80
        )
        enviado = st.form_submit_button("Registrar validação", type="primary")

    if not enviado:
        return None
    if not validado_por.strip():
        st.error("Informe o nome do médico responsável antes de registrar a validação.")
        return None
    return {"validado_por": validado_por.strip(), "parecer": parecer.strip()}
