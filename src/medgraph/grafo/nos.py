"""
[REQ-E1] Nós do fluxo de decisão clínica.

O QUE FAZ:
    Implementa as doze etapas do grafo. Cada nó recebe o estado, faz UMA
    coisa, e devolve um delta.

TODOS OS NÓS SÃO INSTRUMENTADOS:
    O decorador `@instrumentar` registra automaticamente início, fim,
    duração, chaves do estado recebido e delta produzido. A alternativa —
    repetir blocos de log em cada nó — falharia no primeiro esquecimento, e
    um nó sem rastro invalidaria a garantia de auditabilidade exigida pelo
    item 3 do enunciado.

CADA NÓ FAZ UMA COISA SÓ:
    É o que torna o roteamento condicional legível e o que permite testar
    cada etapa isoladamente. `consultar_prontuario` não recupera evidência;
    `raciocinio_clinico` não valida a resposta. Um nó que fizesse as duas
    coisas esconderia, na trilha, qual das duas falhou.

OS NÓS NÃO LEVANTAM EXCEÇÃO PARA FALHA ESPERADA:
    Paciente inexistente, índice ausente, LLM fora do ar — tudo isso vira
    estado, não exceção. O fluxo precisa continuar até o nó de montagem da
    resposta, que sabe transformar qualquer estado numa mensagem útil para o
    médico. Uma exceção no meio do grafo entregaria um traceback.
"""

from __future__ import annotations

from typing import Any

from config.settings import obter_settings
from medgraph.auditoria import TipoEvento, instrumentar, registrar
from medgraph.chains import chain_rag, chain_triagem
from medgraph.grafo.estado import EstadoClinico
from medgraph.guardrails import entrada as guardrail_entrada_mod
from medgraph.guardrails import politicas as mod_politicas
from medgraph.guardrails import regras_clinicas
from medgraph.guardrails import saida as guardrail_saida_mod
from medgraph.llm.provider import obter_llm_com_fallback
from medgraph.logging_config import obter_logger
from medgraph.prontuario.modelos import Paciente
from medgraph.prontuario.repositorio import (
    PacienteNaoEncontradoError,
    RepositorioProntuarios,
)
from medgraph.rag.recuperador import IndiceIndisponivelError, Recuperador, Trecho

log = obter_logger(__name__)

# Recursos caros de inicializar, criados uma vez por processo. O recuperador
# carrega o modelo de embeddings; o repositório abre o banco. Instanciá-los a
# cada consulta faria a latência do fluxo ser dominada por inicialização.
_recuperador: Recuperador | None = None
_repositorio: RepositorioProntuarios | None = None
# Cache do objeto Paciente entre nós da MESMA consulta. Não vai para o estado
# (ver a nota em estado.py sobre não serializar o registro clínico completo).
_paciente_atual: dict[str, Paciente] = {}


def _obter_recuperador() -> Recuperador | None:
    global _recuperador
    if _recuperador is None:
        try:
            _recuperador = Recuperador()
        except IndiceIndisponivelError as exc:
            log.warning("Índice vetorial indisponível: %s", exc)
            return None
    return _recuperador


def _obter_repositorio() -> RepositorioProntuarios | None:
    global _repositorio
    if _repositorio is None:
        try:
            _repositorio = RepositorioProntuarios()
        except FileNotFoundError as exc:
            log.warning("Base de prontuários indisponível: %s", exc)
            return None
    return _repositorio


def _explicar_risco(estado: EstadoClinico) -> str:
    """Descreve, em texto, de onde veio o escore de risco."""
    partes: list[str] = [f"Escore de risco {estado.get('escore_risco')} acima do limiar."]

    achados = estado.get("achados_clinicos", [])
    graves = [a for a in achados if a["severidade"] in ("critica", "alta")]
    if graves:
        partes.append(
            "Achados clínicos: " + "; ".join(a["titulo"] for a in graves) + "."
        )

    gatilhos = estado.get("gatilhos_risco", [])
    if gatilhos:
        partes.append("Gatilhos de contexto: " + ", ".join(gatilhos) + ".")

    if estado.get("intencao_exige_validacao"):
        partes.append(
            "A intenção 'conduta_terapeutica' exige validação por política, "
            "independentemente do escore."
        )
    return " ".join(partes)


def _marca(nome: str) -> dict[str, str]:
    """Entrada de histórico, no formato aceito pelo agregador add_messages."""
    return {"role": "system", "content": nome}


# =============================================================================
# NÓ 1 — GUARDRAIL DE ENTRADA
# =============================================================================
@instrumentar("guardrail_entrada", tipo=TipoEvento.GUARDRAIL)
def no_guardrail_entrada(estado: EstadoClinico) -> EstadoClinico:
    """
    Filtra a pergunta antes de qualquer processamento.  [REQ-3a][REQ-1a]

    Remove identificadores, recusa pedidos fora de escopo e marca situação de
    emergência. Ver `guardrails/entrada.py` para o porquê de filtrar aqui e
    não apenas na saída.
    """
    resultado = guardrail_entrada_mod.verificar(estado.get("pergunta", ""))
    return {
        "aprovado_entrada": resultado.aprovado,
        "pergunta_limpa": resultado.pergunta_limpa,
        "motivo_recusa": resultado.motivo_bloqueio,
        "id_bloqueio": resultado.id_bloqueio,
        "emergencia": resultado.emergencia,
        "termos_emergencia": resultado.termos_emergencia,
        "resposta_final": resultado.resposta_recusa if not resultado.aprovado else "",
        "historico": [_marca("guardrail_entrada")],
    }


# =============================================================================
# NÓ 1b — RECUSA
# =============================================================================
@instrumentar("responder_recusa", tipo=TipoEvento.GUARDRAIL)
def no_responder_recusa(estado: EstadoClinico) -> EstadoClinico:
    """
    Encerra a consulta com a mensagem de recusa.

    A recusa é um DESFECHO legítimo, não um erro. Fica registrada na trilha
    com o identificador da regra que a provocou, o que permite auditar depois
    quantas vezes cada limite foi acionado — informação que entra no relatório
    como evidência de que os guardrails atuam.
    """
    return {
        "resposta_final": estado.get("resposta_final") or "Não posso ajudar com esse pedido.",
        "desfecho": "recusada",
        "historico": [_marca("responder_recusa")],
    }


# =============================================================================
# NÓ 2 — CLASSIFICAÇÃO DE INTENÇÃO
# =============================================================================
@instrumentar("classificar_intencao", tipo=TipoEvento.DECISAO)
def no_classificar_intencao(estado: EstadoClinico) -> EstadoClinico:
    """Decide qual dos cinco caminhos do fluxo seguir. Ver chains/chain_triagem.py."""
    llm = None
    try:
        llm, _ = obter_llm_com_fallback(origem="grafo.classificar_intencao")
    except Exception as exc:
        log.warning("Sem LLM para a triagem (%s) — heurística apenas.", exc)

    resultado = chain_triagem.classificar(
        estado.get("pergunta_limpa") or estado.get("pergunta", ""),
        llm,
        tem_paciente=bool(estado.get("paciente_id")),
    )
    return {
        "intencao": resultado.intencao,
        "metodo_intencao": resultado.metodo,
        "exige_paciente": resultado.exige_paciente,
        "intencao_exige_validacao": resultado.sempre_validacao_humana,
        "historico": [_marca(f"classificar_intencao:{resultado.intencao}")],
    }


# =============================================================================
# NÓ 3 — PRONTUÁRIO
# =============================================================================
@instrumentar("consultar_prontuario", tipo=TipoEvento.BANCO)
def no_consultar_prontuario(estado: EstadoClinico) -> EstadoClinico:
    """
    Carrega o quadro clínico do paciente.  [REQ-2a][REQ-2b]

    O resumo entra no prompt como a fonte [C1] — com o mesmo tratamento de
    citação das outras fontes. Um dado de prontuário é uma afirmação que
    precisa de procedência tanto quanto um trecho de artigo.
    """
    paciente_id = estado.get("paciente_id")
    if not paciente_id:
        return {
            "paciente_encontrado": False,
            "contexto_paciente": "",
            "historico": [_marca("consultar_prontuario:sem_paciente")],
        }

    repositorio = _obter_repositorio()
    if repositorio is None:
        return {
            "paciente_encontrado": False,
            "contexto_paciente": "",
            "erro": "Base de prontuários indisponível.",
            "historico": [_marca("consultar_prontuario:indisponivel")],
        }

    try:
        paciente = repositorio.obter_paciente(paciente_id)
    except PacienteNaoEncontradoError:
        return {
            "paciente_encontrado": False,
            "contexto_paciente": "",
            "erro": f"Paciente '{paciente_id}' não encontrado.",
            "historico": [_marca("consultar_prontuario:nao_encontrado")],
        }

    _paciente_atual[paciente.id] = paciente

    return {
        "paciente_encontrado": True,
        "contexto_paciente": paciente.resumo_clinico(anonimo=True),
        "paciente_resumo": paciente.para_dict(anonimo=True),
        "exames_pendentes": [str(e) for e in paciente.exames_pendentes],
        "historico": [_marca("consultar_prontuario")],
    }


# =============================================================================
# NÓ 4 — RECUPERAÇÃO DE EVIDÊNCIA
# =============================================================================
@instrumentar("recuperar_evidencia", tipo=TipoEvento.RECUPERACAO)
def no_recuperar_evidencia(estado: EstadoClinico) -> EstadoClinico:
    """
    Busca protocolos internos e evidência científica.  [REQ-3c]

    Quando há paciente, o marcador [C1] é reservado ao prontuário, e os
    trechos recuperados recebem [E#] e [P#]. O guardrail de saída usa essa
    lista de marcadores disponíveis para detectar citação inventada.
    """
    recuperador = _obter_recuperador()
    pergunta = estado.get("pergunta_limpa") or estado.get("pergunta", "")

    marcadores: list[str] = []
    if estado.get("paciente_encontrado"):
        marcadores.append("C1")

    if recuperador is None:
        return {
            "trechos": [],
            "marcadores": marcadores,
            "evidencia_suficiente": False,
            "erro": "Índice vetorial indisponível. Rode: make indexar",
            "historico": [_marca("recuperar_evidencia:indisponivel")],
        }

    trechos: list[Trecho] = recuperador.recuperar(pergunta)
    marcadores.extend(t.marcador for t in trechos)

    return {
        "trechos": [t.para_dict() for t in trechos],
        "marcadores": marcadores,
        "evidencia_suficiente": bool(trechos),
        "historico": [_marca(f"recuperar_evidencia:{len(trechos)}")],
    }


# =============================================================================
# NÓ 5 — RACIOCÍNIO CLÍNICO
# =============================================================================
@instrumentar("raciocinio_clinico", tipo=TipoEvento.LLM)
def no_raciocinio_clinico(estado: EstadoClinico) -> EstadoClinico:
    """
    Gera a resposta com a LLM customizada.  [REQ-2]

    É o único nó que chama o modelo para produzir conteúdo clínico. Também é
    o nó de destino do ciclo de reescrita: quando o guardrail de saída
    reprova, o fluxo volta para cá com as instruções de correção anexadas ao
    prompt.
    """
    pol = mod_politicas.carregar()

    try:
        llm, provedor = obter_llm_com_fallback(origem="grafo.raciocinio_clinico")
    except Exception as exc:
        return {
            "resposta_bruta": "",
            "erro": f"Nenhum provedor de LLM disponível: {exc}",
            "historico": [_marca("raciocinio_clinico:sem_llm")],
        }

    # Reconstrói os objetos Trecho a partir do estado serializado.
    trechos = [
        Trecho(
            marcador=d["marcador"], tipo=d["tipo"], titulo=d["titulo"],
            texto=d["texto"], escore=d["escore"],
            identificador=d.get("identificador", ""), secao=d.get("secao", ""),
        )
        for d in estado.get("trechos", [])
    ]

    aviso = pol.texto("alerta_emergencia") if estado.get("emergencia") else ""

    try:
        resposta = chain_rag.responder(
            llm,
            estado.get("pergunta_limpa") or estado.get("pergunta", ""),
            trechos,
            contexto_paciente=estado.get("contexto_paciente", ""),
            intencao=estado.get("intencao", ""),
            instrucoes_correcao=estado.get("instrucoes_correcao", ""),
            aviso_emergencia=aviso,
        )
    except Exception as exc:
        log.exception("Falha na geração da resposta")
        return {
            "resposta_bruta": "",
            "provedor_llm": provedor,
            "erro": f"Falha na geração: {type(exc).__name__}: {exc}",
            "historico": [_marca("raciocinio_clinico:erro")],
        }

    return {
        "resposta_bruta": resposta,
        "provedor_llm": provedor,
        # Limpa as instruções para que uma eventual próxima passagem receba
        # apenas as falhas NOVAS, e não as já corrigidas.
        "instrucoes_correcao": "",
        "historico": [_marca("raciocinio_clinico")],
    }


# =============================================================================
# NÓ 6 — REGRAS CLÍNICAS
# =============================================================================
@instrumentar("regras_clinicas", tipo=TipoEvento.REGRA_CLINICA)
def no_regras_clinicas(estado: EstadoClinico) -> EstadoClinico:
    """
    Verificações determinísticas de segurança.  [REQ-3a]

    Roda sobre a resposta GERADA, e não sobre a pergunta: o que interessa é
    se a conduta sugerida colide com alergia, medicação em uso ou exame
    crítico do paciente. Ver `guardrails/regras_clinicas.py`.
    """
    paciente_id = estado.get("paciente_id")
    paciente = _paciente_atual.get(paciente_id) if paciente_id else None

    resultado = regras_clinicas.verificar(paciente, estado.get("resposta_bruta", ""))
    return {
        "achados_clinicos": [a.para_dict() for a in resultado.achados],
        "risco_clinico": resultado.escore_risco,
        "tem_bloqueio_clinico": resultado.tem_bloqueio,
        "farmacos_detectados": resultado.farmacos_detectados,
        "historico": [_marca(f"regras_clinicas:{len(resultado.achados)}")],
    }


# =============================================================================
# NÓ 7 — GUARDRAIL DE SAÍDA
# =============================================================================
@instrumentar("guardrail_saida", tipo=TipoEvento.GUARDRAIL)
def no_guardrail_saida(estado: EstadoClinico) -> EstadoClinico:
    """Verifica as quatro invariantes da resposta.  [REQ-3a][REQ-3c]"""
    resposta = estado.get("resposta_bruta", "")

    if not resposta:
        return {
            "aprovado_saida": False,
            "falhas_saida": ["resposta_vazia"],
            "instrucoes_correcao": "",
            "historico": [_marca("guardrail_saida:vazia")],
        }

    resultado = guardrail_saida_mod.verificar(
        resposta,
        marcadores_disponiveis=estado.get("marcadores", []),
    )
    return {
        "aprovado_saida": resultado.aprovado,
        "falhas_saida": [f.id for f in resultado.falhas],
        "instrucoes_correcao": resultado.instrucoes_de_correcao,
        "citacoes_usadas": resultado.citacoes,
        "historico": [_marca(f"guardrail_saida:{'ok' if resultado.aprovado else 'reprovado'}")],
    }


# =============================================================================
# NÓ 8 — REESCRITA
# =============================================================================
@instrumentar("reescrever", tipo=TipoEvento.DECISAO)
def no_reescrever(estado: EstadoClinico) -> EstadoClinico:
    """
    Incrementa o contador de tentativas antes de voltar ao raciocínio.

    É um nó separado, e não um incremento dentro do roteamento, porque o
    contador precisa aparecer na trilha como um evento próprio. Ver, na
    auditoria de uma consulta, "reescrever (tentativa 1)" é o que explica por
    que aquela consulta levou o dobro do tempo.
    """
    tentativa = int(estado.get("tentativas_reescrita", 0)) + 1
    registrar(
        TipoEvento.DECISAO,
        f"Reescrita solicitada (tentativa {tentativa})",
        etapa="reescrever",
        falhas=estado.get("falhas_saida", []),
        tentativa=tentativa,
    )
    return {
        "tentativas_reescrita": tentativa,
        "historico": [_marca(f"reescrever:{tentativa}")],
    }


# =============================================================================
# NÓ 8b — DEGRADAÇÃO
# =============================================================================
@instrumentar("degradar_resposta", tipo=TipoEvento.GUARDRAIL)
def no_degradar_resposta(estado: EstadoClinico) -> EstadoClinico:
    """
    Entrega uma resposta mínima segura quando a reescrita se esgota.

    Apresenta as fontes recuperadas sem afirmar nada sobre elas. Degradar em
    vez de falhar é deliberado: o médico prefere receber "não consegui
    sintetizar, aqui estão os trechos" a receber um erro.
    """
    resposta = guardrail_saida_mod.resposta_degradada(
        estado.get("marcadores", []),
        estado.get("trechos", []),
    )
    return {
        "resposta_bruta": resposta,
        "aprovado_saida": True,  # a resposta degradada é segura por construção
        "desfecho": "degradada",
        "historico": [_marca("degradar_resposta")],
    }


# =============================================================================
# NÓ 9 — TRIAGEM DE RISCO
# =============================================================================
@instrumentar("triagem_risco", tipo=TipoEvento.DECISAO)
def no_triagem_risco(estado: EstadoClinico) -> EstadoClinico:
    """
    Decide se a resposta exige validação humana antes de ser entregue.  [REQ-3a]

    COMO O ESCORE É FORMADO:
        Parte do risco calculado pelas regras clínicas (alergia, interação,
        valor crítico) e soma gatilhos de CONTEXTO definidos em
        politicas.yaml — a intenção ser terapêutica, a evidência ser fraca, a
        resposta ter precisado de reescrita.

        A combinação usa o complemento do produto, e não a soma: dois riscos
        de 0,5 resultam em 0,75, não em 1,0. Somar faria qualquer par de
        gatilhos médios saturar o escore e tornar o limiar inútil.

    O ATALHO INCONDICIONAL:
        Quando a política marca a intenção como `sempre_validacao_humana`
        (é o caso de `conduta_terapeutica`), a validação é exigida
        independentemente do escore. Um limiar numérico não deve poder
        dispensar uma regra categórica de segurança.
    """
    cfg = obter_settings()
    pol = mod_politicas.carregar()

    gatilhos: list[str] = []
    risco_ausente = 1.0 - float(estado.get("risco_clinico", 0.0))

    def acionar(identificador: str) -> None:
        nonlocal risco_ausente
        gatilhos.append(identificador)
        risco_ausente *= 1.0 - pol.peso_risco(identificador)

    if estado.get("intencao") == "conduta_terapeutica":
        acionar("intencao_conduta_terapeutica")
    if estado.get("farmacos_detectados"):
        acionar("menciona_medicamento")
    if not estado.get("evidencia_suficiente", True):
        acionar("evidencia_fraca_ou_ausente")
    if int(estado.get("tentativas_reescrita", 0)) > 0:
        acionar("guardrail_reprovou_alguma_vez")

    resumo = estado.get("paciente_resumo") or {}
    if resumo.get("populacao_especial"):
        acionar("paciente_pediatrico_ou_gestante")

    escore = round(1.0 - risco_ausente, 3)

    exige = (
        bool(estado.get("intencao_exige_validacao"))
        or bool(estado.get("tem_bloqueio_clinico"))
        or escore >= cfg.limiar_risco_validacao_humana
    )

    registrar(
        TipoEvento.DECISAO,
        f"Escore de risco {escore} | limiar {cfg.limiar_risco_validacao_humana} | "
        f"validação humana: {'exigida' if exige else 'dispensada'}",
        etapa="triagem_risco",
        escore_risco=escore,
        limiar=cfg.limiar_risco_validacao_humana,
        gatilhos=gatilhos,
        risco_clinico=estado.get("risco_clinico", 0.0),
        exige_validacao=exige,
    )
    return {
        "escore_risco": escore,
        "gatilhos_risco": gatilhos,
        "exige_validacao_humana": exige,
        "historico": [_marca(f"triagem_risco:{escore}")],
    }


# =============================================================================
# NÓ 10 — VALIDAÇÃO HUMANA
# =============================================================================
@instrumentar("aguardar_validacao", tipo=TipoEvento.VALIDACAO_HUMANA)
def no_aguardar_validacao(estado: EstadoClinico) -> EstadoClinico:
    """
    Ponto de parada para validação médica.  [REQ-3a]

    COMO A PAUSA ACONTECE:
        O grafo é compilado com `interrupt_before=["aguardar_validacao"]`. A
        execução para ANTES deste nó rodar, e o estado fica persistido no
        checkpointer. Retomar é chamar o grafo de novo com o mesmo
        `thread_id`, depois de registrar quem validou.

        Este corpo, portanto, só executa DEPOIS da validação — é o nó de
        confirmação, não o de espera.

    É AQUI QUE O REQUISITO MAIS DESTACADO DO ENUNCIADO SE MATERIALIZA:
        "nunca prescrever diretamente, sem validação humana". A frase vira,
        em código, uma execução que fisicamente não continua até que uma
        pessoa registre a validação.
    """
    validador = estado.get("validado_por", "")
    registrar(
        TipoEvento.VALIDACAO_HUMANA,
        f"Validação registrada por {validador or '(não informado)'}",
        etapa="aguardar_validacao",
        validado_por=validador,
        parecer=estado.get("parecer_validacao", ""),
        escore_risco=estado.get("escore_risco"),
    )
    return {
        "historico": [_marca(f"aguardar_validacao:{validador or 'pendente'}")],
    }


# =============================================================================
# NÓ 11 — ALERTAS
# =============================================================================
@instrumentar("emitir_alertas", tipo=TipoEvento.ALERTA)
def no_emitir_alertas(estado: EstadoClinico) -> EstadoClinico:
    """
    Consolida os alertas destinados à equipe médica.

    Não envia nada: registra na trilha de auditoria e devolve no estado, para
    o painel exibir. Integração com o sistema de notificação do hospital
    ficaria aqui — e é justamente por isso que a emissão é um nó separado, e
    não um efeito colateral de outro.
    """
    alertas: list[dict[str, Any]] = []

    if estado.get("emergencia"):
        alertas.append({
            "severidade": "critica",
            "tipo": "emergencia",
            "titulo": "Situação potencialmente crítica identificada",
            "detalhe": (
                "Termos de emergência na consulta: "
                + ", ".join(estado.get("termos_emergencia", []))
            ),
            "acao": "Acionar o Time de Resposta Rápida antes de prosseguir.",
        })

    for achado in estado.get("achados_clinicos", []):
        if achado["severidade"] in ("critica", "alta"):
            alertas.append({
                "severidade": achado["severidade"],
                "tipo": achado["tipo"],
                "titulo": achado["titulo"],
                "detalhe": achado["detalhe"],
                "acao": achado.get("conduta", ""),
            })

    if estado.get("exige_validacao_humana") and not estado.get("validado_por"):
        alertas.append({
            "severidade": "alta",
            "tipo": "validacao_pendente",
            "titulo": "Resposta retida para validação médica",
            # O escore combina DUAS origens: os achados das regras clínicas
            # (alergia, interação, valor crítico) e os gatilhos de contexto
            # (intenção terapêutica, evidência fraca, reescrita). Dizer apenas
            # "gatilhos: nenhum" com risco 0,9 seria enganoso — o risco vinha
            # inteiro do lado clínico.
            "detalhe": _explicar_risco(estado),
            "acao": "Um médico responsável precisa validar antes do uso assistencial.",
        })

    for alerta in alertas:
        # As chaves do alerta são renomeadas antes do espalhamento: 'tipo'
        # colidiria com o parâmetro homônimo de registrar(), e 'nivel' com o
        # nível de log. Espalhar um dicionário de domínio direto na assinatura
        # de uma função é conveniente até o dia em que os dois vocabulários se
        # cruzam — foi o que aconteceu aqui.
        registrar(
            TipoEvento.ALERTA,
            f"[{alerta['severidade'].upper()}] {alerta['titulo']}",
            nivel="WARNING" if alerta["severidade"] in ("critica", "alta") else "INFO",
            etapa="emitir_alertas",
            tipo_alerta=alerta["tipo"],
            severidade=alerta["severidade"],
            detalhe=alerta["detalhe"],
            acao=alerta.get("acao", ""),
        )

    return {
        "alertas": alertas,
        "historico": [_marca(f"emitir_alertas:{len(alertas)}")],
    }


# =============================================================================
# NÓ 12 — MONTAGEM DA RESPOSTA
# =============================================================================
@instrumentar("montar_resposta", tipo=TipoEvento.FIM_ETAPA)
def no_montar_resposta(estado: EstadoClinico) -> EstadoClinico:
    """
    Compõe o texto entregue ao médico.  [REQ-3c]

    Acrescenta, nesta ordem: o aviso de validação pendente (quando houver),
    a resposta, o bloco de alertas, a resolução das citações para as fontes
    completas e o disclaimer.

    A RESOLUÇÃO DAS CITAÇÕES É O QUE FECHA A EXPLAINABILITY:
        A resposta cita [P1]; o bloco de fontes diz o que é [P1] — qual
        protocolo, qual seção, qual escore de similaridade. Sem isso, a
        citação seria um símbolo sem referente.
    """
    pol = mod_politicas.carregar()
    partes: list[str] = []

    pendente = estado.get("exige_validacao_humana") and not estado.get("validado_por")
    if pendente:
        partes.append(f"⚠️ {pol.texto('aviso_validacao_humana')}\n")

    if estado.get("emergencia"):
        partes.append(f"🚨 {pol.texto('alerta_emergencia')}\n")

    partes.append(estado.get("resposta_bruta", "").strip())

    alertas = estado.get("alertas", [])
    if alertas:
        partes.append("\n---\n**Alertas de segurança**\n")
        for alerta in alertas:
            partes.append(
                f"- **[{alerta['severidade'].upper()}] {alerta['titulo']}** — "
                f"{alerta['detalhe']}"
                + (f" _Conduta:_ {alerta['acao']}" if alerta.get("acao") else "")
            )

    # Resolução das citações
    fontes_citadas: list[dict[str, Any]] = []
    citadas = set(estado.get("citacoes_usadas", []))
    for trecho in estado.get("trechos", []):
        if trecho["marcador"] in citadas:
            fontes_citadas.append(trecho)

    if "C1" in citadas and estado.get("paciente_encontrado"):
        fontes_citadas.append({
            "marcador": "C1",
            "tipo": "clinico",
            "titulo": f"Prontuário de {estado.get('paciente_id')}",
            "identificador": str(estado.get("paciente_id")),
            "secao": "resumo clínico",
            "escore": 1.0,
            "texto": estado.get("contexto_paciente", ""),
        })

    if fontes_citadas:
        partes.append("\n---\n**Fontes utilizadas**\n")
        for fonte in sorted(fontes_citadas, key=lambda f: f["marcador"]):
            rotulo = {
                "evidencia": "Evidência científica",
                "protocolo": "Protocolo interno",
                "clinico": "Prontuário do paciente",
            }.get(fonte["tipo"], fonte["tipo"])
            secao = f" · {fonte['secao']}" if fonte.get("secao") else ""
            partes.append(
                f"- **[{fonte['marcador']}]** {rotulo} · `{fonte['identificador']}`{secao}"
                + (f" _(similaridade {fonte['escore']:.2f})_" if fonte.get("escore", 0) < 1 else "")
            )

    partes.append(f"\n---\n_{pol.texto('disclaimer')}_")

    if estado.get("desfecho") == "degradada":
        desfecho = "degradada"
    elif pendente:
        desfecho = "aguardando_validacao"
    elif estado.get("erro"):
        desfecho = "erro"
    else:
        desfecho = "respondida"

    return {
        "resposta_final": "\n".join(partes).strip(),
        "fontes_citadas": fontes_citadas,
        "desfecho": desfecho,
        "historico": [_marca(f"montar_resposta:{desfecho}")],
    }
