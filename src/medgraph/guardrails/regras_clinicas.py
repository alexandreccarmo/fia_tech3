"""
[REQ-3a] Regras clínicas de segurança.

O QUE FAZ:
    Verificações determinísticas, sem LLM, sobre a conduta discutida e os
    dados do paciente: conflito com alergia registrada, interação
    medicamentosa relevante, valor laboratorial crítico, necessidade de ajuste
    por função renal e população especial.

POR QUE ESTAS REGRAS NÃO SÃO DELEGADAS AO MODELO:
    Um LLM pode identificar que ceftriaxona é um betalactâmico. Pode também
    esquecer, alucinar uma classe errada, ou simplesmente não mencionar o
    conflito por não ter dado atenção àquela linha do prontuário. Nenhuma
    dessas falhas seria detectável olhando a resposta.

    A verificação de alergia é o ponto do sistema em que um erro tem
    consequência imediata e grave. Por isso ela é código, não geração: roda
    sempre, do mesmo jeito, é testável, e o resultado entra na trilha de
    auditoria como fato — não como opinião do modelo.

    O modelo redige; as regras verificam. São papéis diferentes.

A DESCOBERTA QUE ORIGINOU ESTE MÓDULO:
    A primeira implementação comparava o nome do fármaco citado com o texto da
    alergia registrada. Falhou no caso que mais importa: um paciente com
    "Penicilina [Betalactâmico]" registrado e uma conduta sugerindo
    "Ceftriaxona 2 g EV" não produzia alerta nenhum — porque as duas palavras
    não se parecem.

    Reatividade cruzada entre fármacos é conhecimento farmacológico, não
    similaridade textual. Daí a tabela CLASSES_DE_FARMACO abaixo: sem ela, a
    regra de segurança mais importante do sistema seria decorativa.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from medgraph.auditoria import TipoEvento, registrar
from medgraph.logging_config import obter_logger
from medgraph.prontuario.modelos import Paciente

log = obter_logger(__name__)


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()


# =============================================================================
# BASE DE CONHECIMENTO FARMACOLÓGICO
# =============================================================================
# Mapeia princípio ativo -> classe farmacológica. É o que permite ligar a
# conduta sugerida ("Ceftriaxona") à alergia registrada ("Penicilina, classe
# Betalactâmico").
#
# ESCOPO E LIMITAÇÃO, DECLARADOS:
#     Esta tabela cobre os fármacos que aparecem nos 15 protocolos internos
#     deste projeto. Não é um formulário terapêutico completo, e num hospital
#     real seria substituída por uma base como a ANVISA, a Micromedex ou o
#     próprio dicionário de medicamentos do prontuário eletrônico.
#
#     A limitação está registrada aqui e no relatório técnico de propósito:
#     um fármaco ausente da tabela NÃO gera alerta, e um sistema clínico que
#     não declara o alcance da sua verificação induz a uma confiança que não
#     merece.
# =============================================================================
CLASSES_DE_FARMACO: Final[dict[str, tuple[str, ...]]] = {
    # Betalactâmicos — a reatividade cruzada entre penicilinas e
    # cefalosporinas é o caso clássico e o mais frequente na prática.
    "betalactamico": (
        "penicilina", "amoxicilina", "ampicilina", "oxacilina", "piperacilina",
        "benzilpenicilina", "cefalexina", "cefazolina", "cefuroxima",
        "ceftriaxona", "cefotaxima", "ceftazidima", "cefepime", "cefalotina",
        "meropenem", "imipenem", "ertapenem", "tazobactam", "clavulanato",
    ),
    "sulfonamida": ("sulfametoxazol", "sulfadiazina", "sulfassalazina", "bactrim"),
    "quinolona": ("ciprofloxacino", "levofloxacino", "moxifloxacino", "norfloxacino"),
    "macrolideo": ("azitromicina", "claritromicina", "eritromicina"),
    "aminoglicosideo": ("gentamicina", "amicacina", "tobramicina", "estreptomicina"),
    "glicopeptideo": ("vancomicina", "teicoplanina"),
    "aine": (
        "ibuprofeno", "diclofenaco", "naproxeno", "cetoprofeno", "nimesulida",
        "piroxicam", "indometacina", "celecoxibe", "acido acetilsalicilico", "aas",
    ),
    "opioide": ("morfina", "fentanil", "tramadol", "codeina", "metadona", "oxicodona"),
    "ieca": ("enalapril", "captopril", "lisinopril", "ramipril", "perindopril"),
    "bra": ("losartana", "valsartana", "candesartana", "olmesartana", "irbesartana"),
    "heparina": ("heparina", "enoxaparina", "dalteparina", "fondaparinux"),
    "contraste_iodado": ("contraste iodado", "iohexol", "iopamidol", "ioversol"),
    "antipsicotico": ("haloperidol", "quetiapina", "risperidona", "olanzapina"),
    "estatina": ("sinvastatina", "atorvastatina", "rosuvastatina", "pravastatina"),
    "anticoagulante_oral": (
        "varfarina", "femprocumona", "rivaroxabana", "apixabana",
        "dabigatrana", "edoxabana",
    ),
    "antiarritmico": ("amiodarona", "propafenona", "sotalol", "digoxina"),
    "antiagregante": ("clopidogrel", "ticagrelor", "prasugrel"),
    "ibp": ("omeprazol", "pantoprazol", "esomeprazol", "lansoprazol"),
    "antidiabetico": ("metformina", "gliclazida", "glibenclamida", "empagliflozina"),
    "diuretico_poupador": ("espironolactona", "amilorida"),
    "diuretico_alca": ("furosemida", "bumetanida"),
    "vasopressor": ("noradrenalina", "adrenalina", "dopamina", "vasopressina", "dobutamina"),
    "corticoide": (
        "prednisona", "prednisolona", "hidrocortisona",
        "dexametasona", "metilprednisolona",
    ),
    "broncodilatador": ("salbutamol", "ipratropio", "formoterol", "budesonida"),
    "antipirético": ("dipirona", "paracetamol"),
}

# Índice inverso: fármaco -> classe. Construído uma vez, na importação.
FARMACO_PARA_CLASSE: Final[dict[str, str]] = {
    _normalizar(farmaco): classe
    for classe, farmacos in CLASSES_DE_FARMACO.items()
    for farmaco in farmacos
}


# =============================================================================
# INTERAÇÕES MEDICAMENTOSAS
# =============================================================================
# Pares com relevância clínica reconhecida, restritos ao que os protocolos
# deste projeto podem sugerir. Mesma limitação de escopo da tabela de classes.
@dataclass(frozen=True)
class Interacao:
    farmaco_a: str
    farmaco_b: str
    gravidade: str          # "grave" | "moderada"
    efeito: str
    conduta: str


INTERACOES: Final[tuple[Interacao, ...]] = (
    Interacao(
        "varfarina", "amiodarona", "grave",
        "A amiodarona inibe o metabolismo da varfarina e eleva o INR de forma acentuada.",
        "Reduzir a dose de varfarina em 30 a 50% e monitorar INR a cada 3 a 5 dias.",
    ),
    Interacao(
        "varfarina", "aine", "grave",
        "Risco hemorrágico somado: anticoagulação com lesão de mucosa gástrica.",
        "Evitar a associação. Se inevitável, associar inibidor de bomba de prótons.",
    ),
    Interacao(
        "varfarina", "sulfonamida", "grave",
        "Sulfametoxazol desloca a varfarina da albumina e inibe seu metabolismo.",
        "Antecipar o controle de INR e considerar antibiótico alternativo.",
    ),
    Interacao(
        "varfarina", "macrolideo", "moderada",
        "Claritromicina e eritromicina elevam o INR.",
        "Monitorar INR em 3 a 5 dias após o início.",
    ),
    Interacao(
        "sinvastatina", "claritromicina", "grave",
        "Inibição do CYP3A4 eleva a concentração da estatina e o risco de rabdomiólise.",
        "Suspender a estatina durante o curso do macrolídeo.",
    ),
    Interacao(
        "ieca", "espironolactona", "grave",
        "Retenção de potássio somada, com risco de hipercalemia grave.",
        "Dosar potássio e creatinina em 48 a 72 horas.",
    ),
    Interacao(
        "bra", "espironolactona", "grave",
        "Retenção de potássio somada, com risco de hipercalemia grave.",
        "Dosar potássio e creatinina em 48 a 72 horas.",
    ),
    Interacao(
        "ieca", "aine", "moderada",
        "Redução da perfusão renal, com risco de lesão renal aguda.",
        "Evitar em paciente com creatinina alterada; monitorar função renal.",
    ),
    Interacao(
        "clopidogrel", "omeprazol", "moderada",
        "O omeprazol reduz a ativação do clopidogrel pelo CYP2C19.",
        "Preferir pantoprazol como inibidor de bomba de prótons.",
    ),
    Interacao(
        "aas", "clopidogrel", "moderada",
        "Dupla antiagregação eleva o risco hemorrágico.",
        "Aceitável quando indicada; reavaliar duração e associar proteção gástrica.",
    ),
    Interacao(
        "aminoglicosideo", "vancomicina", "grave",
        "Nefrotoxicidade somada.",
        "Monitorar creatinina diariamente e dosar nível sérico de vancomicina.",
    ),
    Interacao(
        "metformina", "contraste_iodado", "grave",
        "Risco de acidose láctica se houver deterioração da função renal pelo contraste.",
        "Suspender a metformina no dia do exame e reintroduzir após 48 h, com creatinina normal.",
    ),
)


class Severidade(StrEnum):
    """Gravidade de um achado. Ordena a apresentação e pondera o risco."""

    CRITICA = "critica"
    ALTA = "alta"
    MEDIA = "media"
    INFORMATIVA = "informativa"


PESO_RISCO: Final[dict[Severidade, float]] = {
    Severidade.CRITICA: 0.90,
    Severidade.ALTA: 0.70,
    Severidade.MEDIA: 0.35,
    Severidade.INFORMATIVA: 0.10,
}


@dataclass
class Achado:
    """Um problema de segurança detectado."""

    tipo: str
    severidade: Severidade
    titulo: str
    detalhe: str
    conduta: str = ""
    referencia: str = ""

    def para_dict(self) -> dict[str, str]:
        return {
            "tipo": self.tipo,
            "severidade": self.severidade.value,
            "titulo": self.titulo,
            "detalhe": self.detalhe,
            "conduta": self.conduta,
            "referencia": self.referencia,
        }


@dataclass
class ResultadoVerificacao:
    """Conjunto de achados de uma verificação completa."""

    achados: list[Achado] = field(default_factory=list)
    farmacos_detectados: list[str] = field(default_factory=list)

    @property
    def tem_bloqueio(self) -> bool:
        """Há achado crítico — a resposta não pode ser entregue sem validação."""
        return any(a.severidade is Severidade.CRITICA for a in self.achados)

    @property
    def escore_risco(self) -> float:
        """
        Risco agregado, entre 0 e 1.

        Usamos o complemento do produto das probabilidades de "nenhum
        problema", e não a soma dos pesos: somar faria dois achados médios
        (0,35 cada) somarem 0,70, o mesmo que um achado alto — o que não
        corresponde à intuição clínica. A fórmula abaixo satura suavemente e
        nunca ultrapassa 1.
        """
        risco_ausente = 1.0
        for achado in self.achados:
            risco_ausente *= 1.0 - PESO_RISCO[achado.severidade]
        return round(1.0 - risco_ausente, 3)

    def por_severidade(self, severidade: Severidade) -> list[Achado]:
        return [a for a in self.achados if a.severidade is severidade]

    def para_dict(self) -> dict[str, object]:
        return {
            "total_achados": len(self.achados),
            "escore_risco": self.escore_risco,
            "tem_bloqueio": self.tem_bloqueio,
            "farmacos_detectados": self.farmacos_detectados,
            "achados": [a.para_dict() for a in self.achados],
        }


# =============================================================================
# DETECÇÃO DE FÁRMACOS NO TEXTO
# =============================================================================
def detectar_farmacos(texto: str) -> list[str]:
    """
    Localiza princípios ativos mencionados em um texto livre.

    Percorre a tabela de conhecimento em vez de tentar reconhecer "qualquer
    nome de medicamento" por padrão textual. É mais limitado e muito mais
    previsível: só alerta sobre o que conhece, e nunca inventa um fármaco a
    partir de uma palavra parecida.
    """
    normalizado = _normalizar(texto)
    encontrados: list[str] = []
    for farmaco in FARMACO_PARA_CLASSE:
        if re.search(rf"\b{re.escape(farmaco)}", normalizado):
            encontrados.append(farmaco)

    # Remove o termo mais curto quando ele é parte de um mais longo
    # ("penicilina" dentro de "benzilpenicilina").
    return [
        f for f in encontrados
        if not any(f != outro and f in outro for outro in encontrados)
    ]


def classe_de(farmaco: str) -> str | None:
    """Classe farmacológica de um princípio ativo, se conhecida."""
    normalizado = _normalizar(farmaco)
    if normalizado in FARMACO_PARA_CLASSE:
        return FARMACO_PARA_CLASSE[normalizado]
    for conhecido, classe in FARMACO_PARA_CLASSE.items():
        if conhecido in normalizado:
            return classe
    return None


# =============================================================================
# REGRAS
# =============================================================================
def verificar_alergias(paciente: Paciente, texto_conduta: str) -> list[Achado]:
    """
    Conflito entre a conduta discutida e as alergias registradas.  [REQ-3a]

    Verifica em DOIS níveis:
      1. o próprio fármaco citado consta como alergia;
      2. o fármaco citado pertence à MESMA CLASSE de uma alergia registrada —
         reatividade cruzada.

    O segundo nível é o que dá utilidade real à regra. É também o que exige a
    tabela de conhecimento: "Ceftriaxona" e "Penicilina" não se parecem em
    nada como texto, mas são o mesmo perigo para quem tem alergia a
    betalactâmicos.
    """
    achados: list[Achado] = []
    if not paciente.alergias:
        return achados

    for farmaco in detectar_farmacos(texto_conduta):
        classe_farmaco = classe_de(farmaco)

        for alergia in paciente.alergias:
            substancia_normalizada = _normalizar(alergia.substancia)
            classe_alergia = _normalizar(alergia.classe or "") or classe_de(alergia.substancia)

            direto = substancia_normalizada in farmaco or farmaco in substancia_normalizada
            por_classe = bool(
                classe_farmaco
                and classe_alergia
                and (
                    classe_farmaco == classe_alergia
                    or classe_farmaco in classe_alergia
                    or classe_alergia in classe_farmaco
                )
            )

            if not (direto or por_classe):
                continue

            severidade = (
                Severidade.CRITICA
                if alergia.e_grave or direto
                else Severidade.ALTA
            )
            motivo = (
                f"'{farmaco}' consta diretamente como alergia do paciente."
                if direto
                else (
                    f"'{farmaco}' pertence à classe {classe_farmaco}, a mesma de "
                    f"'{alergia.substancia}', registrada como alergia — há risco de "
                    f"reatividade cruzada."
                )
            )
            achados.append(
                Achado(
                    tipo="alergia",
                    severidade=severidade,
                    # O título nomeia o par (fármaco citado × alergia registrada).
                    # Quando a conduta menciona vários fármacos da mesma classe,
                    # cada um gera seu próprio achado — e o médico precisa saber
                    # QUAL deles conflita, não apenas que há um conflito.
                    titulo=f"Conflito: {farmaco} × alergia a {alergia.substancia}",
                    detalhe=(
                        f"{motivo} Gravidade registrada: {alergia.gravidade or 'não informada'}"
                        + (f". Reação: {alergia.reacao}" if alergia.reacao else "")
                    ),
                    conduta="Escolher alternativa de outra classe e confirmar a alergia com o paciente.",
                    referencia=f"prontuário de {paciente.id}",
                )
            )
    return achados


def verificar_interacoes(paciente: Paciente, texto_conduta: str) -> list[Achado]:
    """
    Interação entre o que está sendo sugerido e o que o paciente já usa.

    Compara os fármacos citados na conduta contra as medicações ATIVAS do
    prontuário. Interações entre dois fármacos que o paciente já usa não são
    reportadas aqui: já foram avaliadas por quem prescreveu, e repeti-las a
    cada consulta produziria ruído que faria o médico ignorar também os
    alertas novos.
    """
    achados: list[Achado] = []
    citados = detectar_farmacos(texto_conduta)
    if not citados:
        return achados

    em_uso = [_normalizar(m.principio_ativo) for m in paciente.medicacoes_ativas]

    def corresponde(termo: str, farmacos: Iterable[str]) -> str | None:
        """Casa por nome exato ou por classe."""
        for farmaco in farmacos:
            if termo in farmaco or farmaco in termo:
                return farmaco
            if classe_de(farmaco) == termo:
                return farmaco
        return None

    for interacao in INTERACOES:
        # A interação vale nos dois sentidos: o fármaco citado pode ser
        # qualquer um dos dois lados do par.
        for termo_citado, termo_em_uso in (
            (interacao.farmaco_a, interacao.farmaco_b),
            (interacao.farmaco_b, interacao.farmaco_a),
        ):
            novo = corresponde(termo_citado, citados)
            antigo = corresponde(termo_em_uso, em_uso)
            if not (novo and antigo):
                continue

            achados.append(
                Achado(
                    tipo="interacao",
                    severidade=(
                        Severidade.ALTA if interacao.gravidade == "grave" else Severidade.MEDIA
                    ),
                    titulo=f"Interação medicamentosa: {novo} + {antigo}",
                    detalhe=interacao.efeito,
                    conduta=interacao.conduta,
                    referencia=f"medicações ativas de {paciente.id}",
                )
            )
            break  # não reportar o mesmo par duas vezes

    return achados


def verificar_valores_criticos(paciente: Paciente) -> list[Achado]:
    """Resultados laboratoriais em faixa crítica."""
    return [
        Achado(
            tipo="valor_critico",
            severidade=Severidade.CRITICA,
            titulo=f"Valor crítico: {exame.nome}",
            detalhe=str(exame),
            conduta="Confirmar o resultado e definir conduta imediata antes de qualquer outra medida.",
            referencia=f"exames de {paciente.id}",
        )
        for exame in paciente.exames_criticos
    ]


def verificar_funcao_renal(paciente: Paciente, texto_conduta: str) -> list[Achado]:
    """
    Necessidade de ajuste de dose por função renal.

    Não calcula o clearance nem sugere a dose: apenas SINALIZA que o paciente
    tem creatinina alterada e que a conduta menciona um fármaco de eliminação
    renal. Calcular a dose seria prescrever, o que o assistente não faz.
    """
    creatininas = [
        e for e in paciente.exames
        if "creatinina" in _normalizar(e.nome) and e.valor is not None and e.fora_da_faixa
    ]
    if not creatininas:
        return []

    # Fármacos com eliminação predominantemente renal, entre os que a tabela
    # de conhecimento reconhece.
    de_eliminacao_renal = {
        "vancomicina", "gentamicina", "amicacina", "tobramicina", "enoxaparina",
        "metformina", "ciprofloxacino", "levofloxacino", "meropenem", "cefepime",
        "piperacilina", "fondaparinux",
    }
    citados = [f for f in detectar_farmacos(texto_conduta) if f in de_eliminacao_renal]
    if not citados:
        return []

    pior = max(creatininas, key=lambda e: e.valor or 0)
    return [
        Achado(
            tipo="funcao_renal",
            severidade=Severidade.ALTA,
            titulo="Ajuste de dose por função renal",
            detalhe=(
                f"Paciente com {pior}. A conduta menciona {', '.join(citados)}, "
                f"de eliminação predominantemente renal."
            ),
            conduta="Calcular o clearance e ajustar a dose conforme o protocolo do fármaco.",
            referencia=f"exames de {paciente.id}",
        )
    ]


def verificar_populacao_especial(paciente: Paciente) -> list[Achado]:
    """Gestante, paciente pediátrico ou idoso acima de 80 anos."""
    if not paciente.populacao_especial:
        return []

    quais: list[str] = []
    if paciente.gestante:
        quais.append("gestante")
    if paciente.pediatrico:
        quais.append(f"pediátrico ({paciente.idade} anos)")
    if paciente.idade >= 80:
        quais.append(f"idoso ({paciente.idade} anos)")

    return [
        Achado(
            tipo="populacao_especial",
            severidade=Severidade.MEDIA,
            titulo="População especial",
            detalhe=(
                f"Paciente {', '.join(quais)}. Doses, contraindicações e metas "
                f"terapêuticas podem diferir do adulto padrão."
            ),
            conduta="Conferir a seção de população especial do protocolo aplicável.",
            referencia=f"prontuário de {paciente.id}",
        )
    ]


# =============================================================================
# VERIFICAÇÃO COMPLETA
# =============================================================================
def verificar(
    paciente: Paciente | None,
    texto_conduta: str,
    *,
    regras: Sequence[str] | None = None,
) -> ResultadoVerificacao:
    """
    Roda todas as regras aplicáveis e devolve os achados consolidados.

    Args:
        paciente: None quando a pergunta é conceitual, sem paciente vinculado.
            Nesse caso, nenhuma regra que dependa do prontuário se aplica.
        regras: subconjunto de regras a executar. Usado pelos testes.
    """
    resultado = ResultadoVerificacao()
    resultado.farmacos_detectados = detectar_farmacos(texto_conduta)

    if paciente is None:
        registrar(
            TipoEvento.REGRA_CLINICA,
            "Verificação sem paciente vinculado — regras de prontuário não se aplicam",
            farmacos_detectados=resultado.farmacos_detectados,
        )
        return resultado

    todas = {
        "alergias": lambda: verificar_alergias(paciente, texto_conduta),
        "interacoes": lambda: verificar_interacoes(paciente, texto_conduta),
        "valores_criticos": lambda: verificar_valores_criticos(paciente),
        "funcao_renal": lambda: verificar_funcao_renal(paciente, texto_conduta),
        "populacao_especial": lambda: verificar_populacao_especial(paciente),
    }
    selecionadas = regras or list(todas)

    for nome in selecionadas:
        resultado.achados.extend(todas[nome]())

    # Mais grave primeiro: é a ordem em que o médico precisa ler.
    ordem = {s: i for i, s in enumerate(Severidade)}
    resultado.achados.sort(key=lambda a: ordem[a.severidade])

    registrar(
        TipoEvento.REGRA_CLINICA,
        f"{len(resultado.achados)} achado(s) de segurança | risco {resultado.escore_risco}",
        nivel="WARNING" if resultado.tem_bloqueio else "INFO",
        paciente_id=paciente.id,
        farmacos_detectados=resultado.farmacos_detectados,
        escore_risco=resultado.escore_risco,
        tem_bloqueio=resultado.tem_bloqueio,
        achados=[f"{a.severidade.value}:{a.titulo}" for a in resultado.achados],
    )
    return resultado
