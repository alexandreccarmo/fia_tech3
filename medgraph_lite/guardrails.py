"""
Limites de atuacao do assistente.

O item 3 do enunciado pede tres coisas: limites para evitar sugestoes
improprias, logging para auditoria e explainability por citacao de fonte. As
verificacoes daqui cobrem a primeira e a terceira; o logging fica no grafo.

O principio: o assistente nunca prescreve. Ele mostra evidencia, cita a fonte e
devolve a decisao ao medico.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Familias de farmacos que compartilham reatividade cruzada. Sem esta tabela, a
# verificacao de alergia seria comparacao de texto - e "Penicilina" nao casa com
# "Ceftriaxona", embora as duas sejam betalactamicos.
CLASSES = {
    "penicilina": "betalactamico",
    "amoxicilina": "betalactamico",
    "ampicilina": "betalactamico",
    "ceftriaxona": "betalactamico",
    "cefepime": "betalactamico",
    "cefazolina": "betalactamico",
    "meropenem": "betalactamico",
    "azitromicina": "macrolideo",
    "claritromicina": "macrolideo",
    "levofloxacino": "quinolona",
    "ciprofloxacino": "quinolona",
    "vancomicina": "glicopeptideo",
    "varfarina": "anticoagulante",
    "amiodarona": "antiarritmico",
    "sulfametoxazol": "sulfonamida",
    "fluconazol": "antifungico",
}

# Pares com interacao de relevancia clinica reconhecida.
INTERACOES = [
    ("varfarina", "amiodarona", "Amiodarona eleva o INR da varfarina"),
    ("varfarina", "sulfametoxazol", "Sulfametoxazol potencializa a varfarina"),
    ("varfarina", "fluconazol", "Fluconazol eleva o INR da varfarina"),
]

# Pedidos que o assistente recusa, por estarem fora do que ele pode fazer.
PEDIDOS_BLOQUEADOS = [
    "pule a validacao",
    "sem validacao humana",
    "prescreva direto",
    "me de a receita",
    "assine o atestado",
    "ignore o protocolo",
]

# Termos que indicam que o farmaco esta sendo EVITADO, e nao sugerido.
EVITACAO = [
    "evitar", "evite", "contraindicad", "nao usar", "nao administrar",
    "alergia", "alergic", "anafilaxia", "suspender", "risco de",
]


def _normalizar(texto: str) -> str:
    """Tira acento e caixa: 'validação' e 'validacao' viram a mesma coisa."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


@dataclass
class Achado:
    severidade: str  # "critico" | "atencao" | "informativo"
    mensagem: str


@dataclass
class Verificacao:
    aprovado: bool
    achados: list[Achado] = field(default_factory=list)

    @property
    def criticos(self) -> list[Achado]:
        return [a for a in self.achados if a.severidade == "critico"]


def verificar_entrada(pergunta: str) -> Verificacao:
    """Recusa pedidos que estao fora do que o assistente pode fazer."""
    normalizada = _normalizar(pergunta)
    for bloqueado in PEDIDOS_BLOQUEADOS:
        if _normalizar(bloqueado) in normalizada:
            return Verificacao(
                aprovado=False,
                achados=[Achado("critico", f"Pedido fora do escopo: '{bloqueado}'")],
            )
    return Verificacao(aprovado=True)


def _mencao_em_contexto_de_evitacao(texto: str, farmaco: str) -> bool:
    """
    A janela e a FRASE, e nao um numero de caracteres ao redor.

    Em "Evitar penicilina. Iniciar ceftriaxona", uma janela por proximidade
    alcancaria o "evitar" da frase anterior e classificaria a ceftriaxona como
    evitada - errando na direcao mais perigosa.
    """
    normalizado = _normalizar(texto)
    for frase in re.split(r"[.;!?\n]", normalizado):
        if farmaco in frase:
            return any(termo in frase for termo in EVITACAO)
    return False


def detectar_farmacos(texto: str) -> list[str]:
    normalizado = _normalizar(texto)
    return [f for f in CLASSES if f in normalizado]


def verificar_resposta(resposta: str, paciente=None,
                       marcadores_recuperados=None) -> Verificacao:
    """
    Confere a resposta contra as regras clinicas e o contrato de citacao.

    Cada achado tem severidade propria: um conflito de alergia e critico, uma
    mencao ao farmaco em contexto de evitacao e apenas informativa. Rebaixar em
    vez de suprimir importa - se a heuristica errar, o pior que acontece e um
    alerta discreto onde deveria haver um grave.

    `marcadores_recuperados` sao os identificadores das fontes que o RAG
    entregou nesta consulta. Com eles, a citacao deixa de ser conferida apenas
    quanto a existencia e passa a ser conferida quanto a procedencia.
    """
    achados: list[Achado] = []
    farmacos = detectar_farmacos(resposta)

    if paciente is not None and "nenhuma" not in paciente.alergias.lower():
        alergicos = detectar_farmacos(paciente.alergias)
        classes_proibidas = {CLASSES[f] for f in alergicos}
        for farmaco in farmacos:
            if CLASSES[farmaco] not in classes_proibidas:
                continue
            if _mencao_em_contexto_de_evitacao(resposta, farmaco):
                achados.append(Achado(
                    "informativo",
                    f"{farmaco.capitalize()} citado em contexto de evitacao",
                ))
            else:
                achados.append(Achado(
                    "critico",
                    f"{farmaco.capitalize()} e {CLASSES[farmaco]}, mesma classe da "
                    f"alergia registrada ({paciente.alergias})",
                ))

    if paciente is not None:
        em_uso = detectar_farmacos(paciente.medicacoes)
        for a, b, motivo in INTERACOES:
            if (a in em_uso and b in farmacos) or (b in em_uso and a in farmacos):
                achados.append(Achado("critico", f"Interacao: {motivo}"))

    achados.extend(_conferir_citacao(resposta, marcadores_recuperados))

    return Verificacao(aprovado=not any(a.severidade == "critico" for a in achados),
                       achados=achados)


def _conferir_citacao(resposta: str, marcadores_recuperados=None) -> list[Achado]:
    """
    Explainability em duas partes: existir a citacao, e a citacao ser real.

    Conferir apenas que HA um marcador deixa passar a falha mais provavel de um
    modelo ajustado a citar: inventar o marcador. O treino ensina o formato
    "... [E1]" com os exemplos do PubMedQA, e nada impede que o modelo escreva
    [E1] respondendo sobre um protocolo do hospital - o formato fica perfeito e
    a fonte nao existe.

    Por isso a segunda parte compara o que foi citado com o que foi de fato
    recuperado. Quando quem chama nao informa os marcadores (uso avulso da
    funcao, fora do grafo), so a primeira parte se aplica.
    """
    citados = set(re.findall(r"\[([A-Z]+\d+)\]", resposta))
    if not citados:
        return [Achado("critico", "Resposta sem citacao de fonte")]

    if marcadores_recuperados is None:
        return []

    validos = set(marcadores_recuperados)
    inventados = sorted(citados - validos)
    if not inventados:
        return []
    if citados & validos:
        # Ha ao menos uma fonte real: o excedente e ruido, e nao invencao pura.
        return [Achado("atencao",
                       f"Citacao a fonte nao recuperada: {', '.join(inventados)}")]
    return [Achado("critico",
                   f"Resposta cita fonte que nao foi recuperada: "
                   f"{', '.join(inventados)}")]


def verificar_prontuario(paciente=None) -> Verificacao:
    """
    Le o prontuario e produz alertas, antes de qualquer resposta do modelo.

    Esta verificacao nao olha o texto gerado - ela olha o paciente. O enunciado
    pede que o fluxo "verifique exames pendentes" ao receber informacoes sobre
    o paciente, e isso acontece independentemente do que a LLM venha a
    responder. Estava dentro de `verificar_resposta`, onde dependia de haver
    resposta para ser feita.
    """
    if paciente is None:
        return Verificacao(aprovado=True)

    achados = [Achado("atencao", f"Exame critico: {exame}")
               for exame in paciente.exames_criticos]
    achados += [Achado("atencao", f"Exame pendente: {exame}")
                for exame in paciente.exames_pendentes]
    return Verificacao(aprovado=True, achados=achados)


DISCLAIMER = (
    "\n\n---\nApoio a decisao clinica. Nao substitui avaliacao medica: "
    "a conduta final e do medico responsavel."
)
