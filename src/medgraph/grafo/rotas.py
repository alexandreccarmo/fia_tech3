"""
[REQ-E1] Roteamento condicional do fluxo.

O QUE FAZ:
    As funções que decidem, a cada bifurcação, qual será o próximo nó.

POR QUE AS ROTAS FICAM SEPARADAS DOS NÓS:
    Um nó transforma estado; uma rota lê estado e escolhe um caminho. Manter
    as duas coisas separadas deixa o desenho do fluxo legível em um arquivo
    só — dá para entender todas as decisões do grafo lendo estas cinquenta
    linhas, sem abrir a implementação de nenhum nó.

    Também torna as rotas testáveis com dicionários simples, sem precisar
    executar o fluxo.

TODA ROTA REGISTRA A DECISÃO NA TRILHA:
    Saber que o fluxo passou por `reescrever` é útil; saber POR QUE ele foi
    para lá é o que permite auditar. Cada função abaixo registra o critério
    que aplicou junto com o destino escolhido.
"""

from __future__ import annotations

from config.settings import obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.grafo.estado import EstadoClinico
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


def _decidir(destino: str, criterio: str, **dados) -> str:
    registrar(
        TipoEvento.DECISAO,
        f"→ {destino} ({criterio})",
        etapa="roteamento",
        destino=destino,
        criterio=criterio,
        **dados,
    )
    return destino


def apos_guardrail_entrada(estado: EstadoClinico) -> str:
    """Pedido bloqueado encerra o fluxo; aprovado segue para a triagem."""
    if not estado.get("aprovado_entrada", False):
        return _decidir(
            "responder_recusa",
            "entrada bloqueada pelo guardrail",
            id_bloqueio=estado.get("id_bloqueio", ""),
        )
    return _decidir("classificar_intencao", "entrada aprovada")


def apos_classificar_intencao(estado: EstadoClinico) -> str:
    """
    Só consulta o prontuário quando há paciente E a intenção o exige.

    Uma dúvida conceitual sobre sepse não precisa de prontuário, e carregá-lo
    seria acesso a dado de paciente sem justificativa assistencial — algo que
    ficaria registrado na trilha de auditoria como um acesso indevido.
    """
    intencao = estado.get("intencao", "")
    tem_paciente = bool(estado.get("paciente_id"))

    # Conduta terapêutica com paciente vinculado sempre passa pelo prontuário,
    # mesmo que a política não marque `exige_paciente`: sugerir conduta sem
    # olhar as alergias do paciente que está na frente do médico seria pior do
    # que não sugerir nada.
    precisa = tem_paciente and (
        estado.get("exige_paciente", False) or intencao == "conduta_terapeutica"
    )

    if precisa:
        return _decidir(
            "consultar_prontuario", "intenção exige dados do paciente", intencao=intencao
        )
    return _decidir(
        "recuperar_evidencia",
        "pergunta conceitual ou sem paciente vinculado",
        intencao=intencao,
        tem_paciente=tem_paciente,
    )


def apos_guardrail_saida(estado: EstadoClinico) -> str:
    """
    O ciclo de reescrita — a única aresta que volta no grafo.

    Três saídas:
      aprovada           segue para a triagem de risco;
      reprovada, com     volta ao raciocínio com as instruções de correção;
        tentativas
      reprovada, sem     degrada para a resposta mínima segura.
        tentativas

    O TETO DE TENTATIVAS NÃO É NEGOCIÁVEL:
        Sem ele, uma resposta que o modelo não consegue corrigir — porque não
        há evidência suficiente, por exemplo — faria o grafo girar
        indefinidamente, consumindo tempo e tokens. O limite vem do .env
        (MAX_TENTATIVAS_GUARDRAIL), com teto também declarado em
        politicas.yaml.
    """
    cfg = obter_settings()

    if estado.get("aprovado_saida", False):
        return _decidir("triagem_risco", "resposta aprovada no guardrail de saída")

    tentativas = int(estado.get("tentativas_reescrita", 0))
    if tentativas < cfg.max_tentativas_guardrail:
        return _decidir(
            "reescrever",
            f"reprovada, tentativa {tentativas + 1} de {cfg.max_tentativas_guardrail}",
            falhas=estado.get("falhas_saida", []),
        )

    return _decidir(
        "degradar_resposta",
        f"tentativas esgotadas ({cfg.max_tentativas_guardrail})",
        falhas=estado.get("falhas_saida", []),
    )


def apos_triagem_risco(estado: EstadoClinico) -> str:
    """
    Alto risco pausa o fluxo para validação médica.  [REQ-3a]

    A bifurcação acontece DEPOIS da emissão dos alertas, não antes. O médico
    que valida precisa ver quais conflitos de segurança foram detectados —
    validar sem essa informação produziria um registro de aprovação sem
    fundamento.

    É a materialização do requisito mais destacado do enunciado. Quando esta
    rota escolhe `aguardar_validacao`, a execução do grafo PARA — não é uma
    marcação no texto, é o fluxo que fisicamente não avança até que uma
    pessoa registre a validação.
    """
    if estado.get("exige_validacao_humana", False) and not estado.get("validado_por"):
        return _decidir(
            "aguardar_validacao",
            "risco acima do limiar ou intenção de conduta terapêutica",
            escore_risco=estado.get("escore_risco"),
            gatilhos=estado.get("gatilhos_risco", []),
        )
    return _decidir(
        "montar_resposta",
        "risco abaixo do limiar",
        escore_risco=estado.get("escore_risco"),
    )
