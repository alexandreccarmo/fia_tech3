"""
[REQ-3a] Guardrail de entrada.

O QUE FAZ:
    Examina a pergunta ANTES de qualquer processamento: remove identificadores,
    verifica se o pedido está dentro do escopo do assistente e detecta menção a
    situação de emergência.

POR QUE FILTRAR NA ENTRADA, E NÃO SÓ NA SAÍDA:
    Três razões, em ordem de importância.

    1. SEGURANÇA. Um pedido para "prescrever direto, sem validação" não deve
       nem chegar ao modelo. Deixar o modelo processá-lo e depois barrar a
       resposta significa confiar que o guardrail de saída vai reconhecer o
       resultado — e um modelo instruído a burlar as regras pode produzir algo
       que passe pela verificação de forma.

    2. PRIVACIDADE. Se o médico colar um trecho de prontuário com o nome do
       paciente, esse nome não pode entrar no prompt, nem no índice de cache,
       nem na trilha de auditoria. A anonimização aqui é a primeira e mais
       barata linha de defesa.

    3. CUSTO E LATÊNCIA. Recusar na entrada custa milissegundos; recusar na
       saída custa uma inferência inteira.

EMERGÊNCIA NÃO É BLOQUEIO:
    Menção a parada cardiorrespiratória ou choque anafilático não impede a
    resposta. O que ela faz é antepor a orientação de acionar o time de
    resposta rápida. Bloquear seria pior do que inútil: o médico está no meio
    de uma emergência e precisa da informação, não de uma recusa.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.dados.anonimizador import Anonimizador, Politica
from medgraph.guardrails import politicas as mod_politicas
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


def _sem_acento(texto: str) -> str:
    """Remove acentos, preservando o restante do texto."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


@dataclass
class ResultadoEntrada:
    """O veredito do guardrail de entrada."""

    aprovado: bool
    pergunta_limpa: str
    motivo_bloqueio: str = ""
    id_bloqueio: str = ""
    resposta_recusa: str = ""
    emergencia: bool = False
    termos_emergencia: list[str] = field(default_factory=list)
    identificadores_removidos: dict[str, int] = field(default_factory=dict)

    def para_dict(self) -> dict[str, Any]:
        return {
            "aprovado": self.aprovado,
            "id_bloqueio": self.id_bloqueio,
            "motivo_bloqueio": self.motivo_bloqueio,
            "emergencia": self.emergencia,
            "termos_emergencia": self.termos_emergencia,
            "identificadores_removidos": self.identificadores_removidos,
        }


def verificar(
    pergunta: str,
    *,
    nomes_conhecidos: list[str] | None = None,
    cfg: Settings | None = None,
) -> ResultadoEntrada:
    """
    Aplica o guardrail de entrada.

    A ORDEM DAS VERIFICAÇÕES É DELIBERADA:
        1. tamanho — barato, e entrada gigante costuma ser tentativa de
           injeção ou colagem acidental de prontuário inteiro;
        2. padrões de bloqueio — sobre o texto ORIGINAL, porque anonimizar
           antes poderia destruir justamente a frase que se quer detectar;
        3. anonimização — só o que sobreviveu chega ao modelo;
        4. emergência — não bloqueia, apenas marca.
    """
    cfg = cfg or obter_settings()
    pol = mod_politicas.carregar()

    # --- 1. Tamanho ---------------------------------------------------------
    limite = int(pol.entrada.get("max_caracteres_pergunta", 4000))
    if len(pergunta) > limite:
        resultado = ResultadoEntrada(
            aprovado=False,
            pergunta_limpa="",
            id_bloqueio="tamanho_excessivo",
            motivo_bloqueio=f"A pergunta tem {len(pergunta)} caracteres; o limite é {limite}.",
            resposta_recusa=(
                f"A pergunta excede o limite de {limite} caracteres. "
                f"Reformule de forma mais objetiva ou informe o identificador do "
                f"paciente em vez de colar o prontuário."
            ),
        )
        registrar(
            TipoEvento.GUARDRAIL, "Entrada bloqueada: tamanho excessivo",
            nivel="WARNING", etapa="guardrail_entrada", **resultado.para_dict(),
        )
        return resultado

    # --- 2. Padrões de bloqueio, sobre o texto original --------------------
    #
    # A comparação é feita sobre uma versão SEM ACENTO da pergunta, e os
    # padrões em politicas.yaml são escritos em ASCII.
    #
    # Foi uma lacuna real de segurança: o padrão `burlar_validacao` procurava
    # por "validacao" e o pedido "Pule a validação humana" — escrito como
    # qualquer pessoa escreveria — passava direto pelo guardrail. Escrever
    # cada padrão com as duas grafias ("valida[çc][ãa]o") funcionaria, mas
    # teria que ser lembrado em todo padrão novo, e o esquecimento seria
    # silencioso. Normalizar o texto resolve de uma vez, para todos.
    pergunta_sem_acento = _sem_acento(pergunta)
    for padrao in pol.padroes_bloqueio:
        if padrao.regex.search(pergunta_sem_acento) or padrao.regex.search(pergunta):
            chave_texto = {
                "prescricao_direta_ao_paciente": "recusa_prescricao_direta",
                "burlar_validacao": "recusa_prescricao_direta",
                "exportacao_em_massa": "recusa_dados_em_massa",
                "ignorar_instrucoes": "recusa_fora_de_escopo",
            }.get(padrao.id, "recusa_fora_de_escopo")

            resultado = ResultadoEntrada(
                aprovado=False,
                pergunta_limpa="",
                id_bloqueio=padrao.id,
                motivo_bloqueio=padrao.motivo,
                resposta_recusa=pol.texto(chave_texto),
            )
            registrar(
                TipoEvento.GUARDRAIL,
                f"Entrada bloqueada: {padrao.id}",
                nivel="WARNING",
                etapa="guardrail_entrada",
                **resultado.para_dict(),
            )
            return resultado

    # --- 3. Anonimização ----------------------------------------------------
    anonimizador = Anonimizador(
        politica=Politica.MASCARAR, nomes_conhecidos=nomes_conhecidos or []
    )
    limpa = anonimizador.redigir(pergunta)

    # --- 4. Emergência (marca, não bloqueia) --------------------------------
    # Mesma normalização dos padrões de bloqueio: "sepse grave" e "AVC agudo"
    # aparecem com e sem acento nos termos configurados.
    minuscula = _sem_acento(limpa.lower())
    encontrados = [t for t in pol.termos_emergencia if _sem_acento(t) in minuscula]

    resultado = ResultadoEntrada(
        aprovado=True,
        pergunta_limpa=limpa,
        emergencia=bool(encontrados),
        termos_emergencia=encontrados,
        identificadores_removidos=anonimizador.estatisticas(),
    )

    if anonimizador.total_removido:
        registrar(
            TipoEvento.ANONIMIZACAO,
            f"{anonimizador.total_removido} identificador(es) removido(s) da pergunta",
            etapa="guardrail_entrada",
            por_tipo=anonimizador.estatisticas(),
        )

    if encontrados:
        registrar(
            TipoEvento.ALERTA,
            f"Situação potencialmente crítica identificada: {', '.join(encontrados)}",
            nivel="WARNING",
            etapa="guardrail_entrada",
            termos=encontrados,
        )

    registrar(
        TipoEvento.GUARDRAIL,
        "Entrada aprovada",
        etapa="guardrail_entrada",
        emergencia=resultado.emergencia,
        identificadores_removidos=anonimizador.total_removido,
    )
    return resultado
