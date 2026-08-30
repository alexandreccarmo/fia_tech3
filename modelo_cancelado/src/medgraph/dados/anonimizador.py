"""
[REQ-1a] Anonimizacao de dados pessoais e de saude.

O QUE FAZ:
    Encontra e substitui informacao identificavel em texto livre - CPF, RG,
    Cartao Nacional de Saude, telefone, e-mail, CEP, data de nascimento,
    numero de prontuario, CRM, nome de pessoa e idade extrema.

POR QUE EXISTE:
    O item 1 do enunciado exige preparar os dados "com tecnicas de
    preprocessing, anonimizacao e curadoria". Mas a razao real e mais forte
    do que o requisito: um assistente clinico manipula PHI (Protected Health
    Information) em tres momentos perigosos, e todos os tres passam por aqui:

      1. ANTES DO FINE-TUNING - o que entra no treino fica gravado nos pesos
         do modelo e pode ser regurgitado depois. Este e o vazamento mais
         dificil de reverter, porque nao ha como "apagar" um dado de dentro
         de uma rede neural.
      2. ANTES DA AUDITORIA - a trilha JSONL e gravada em disco e lida pelo
         painel. Sem redacao, ela vira um banco paralelo de dados pessoais.
      3. ANTES DA RESPOSTA AO MEDICO - o guardrail de saida usa este modulo
         para garantir que nenhum identificador vazou do prontuario para o
         texto gerado.

DUAS POLITICAS DE SUBSTITUICAO:

    MASCARAR (padrao para logs e auditoria)
        "Maria Silva, CPF 123.456.789-00" -> "[NOME], CPF [CPF]"
        Simples e irreversivel. Perde a informacao de que duas mencoes se
        referem a mesma pessoa.

    PSEUDONIMIZAR (padrao para o dataset de fine-tuning)
        "Maria Silva ... a paciente Maria" -> "[NOME_7c1a] ... a paciente [NOME_7c1a]"
        O mesmo valor sempre vira o mesmo token, porque o sufixo e um HMAC
        do valor com uma chave secreta. Isso preserva a COERENCIA REFERENCIAL
        do texto - o modelo continua aprendendo que o sujeito e o mesmo ao
        longo do documento - sem revelar quem e. E irreversivel sem a chave.

O QUE NAO E ANONIMIZADO, DE PROPOSITO:
    Valores laboratoriais, doses, sinais vitais e datas de exame. Sao dados
    clinicos, nao identificadores, e apaga-los destruiria exatamente a
    informacao que o assistente precisa para raciocinar. Ha teste garantindo
    que "potassio 6.8 mEq/L" e "Ceftriaxona 2 g EV" sobrevivem intactos.

COMO USAR:
    from medgraph.dados.anonimizador import Anonimizador, Politica

    anon = Anonimizador(politica=Politica.PSEUDONIMIZAR)
    limpo = anon.redigir("Paciente Maria Silva, CPF 123.456.789-00")
    achados = anon.analisar(texto)      # sem alterar o texto
    print(anon.estatisticas())
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from re import Pattern
from typing import Final

from medgraph.auditoria import TipoEvento, registrar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)


class TipoPII(StrEnum):
    """Categorias de informacao identificavel reconhecidas pelo modulo."""

    CPF = "cpf"
    RG = "rg"
    CNS = "cartao_sus"
    TELEFONE = "telefone"
    EMAIL = "email"
    CEP = "cep"
    DATA_NASCIMENTO = "data_nascimento"
    PRONTUARIO = "prontuario"
    CRM = "crm"
    NOME = "nome"
    IDADE_EXTREMA = "idade_extrema"


class Politica(StrEnum):
    """Como o valor encontrado deve ser substituido."""

    MASCARAR = "mascarar"
    PSEUDONIMIZAR = "pseudonimizar"


# -----------------------------------------------------------------------------
# PADROES
# -----------------------------------------------------------------------------
# A ORDEM IMPORTA. Os padroes sao aplicados em sequencia e o texto ja
# substituido nao e reexaminado. Por isso os mais especificos vem primeiro:
# o CNS (15 digitos) precisa ser consumido antes do telefone, e o CPF antes
# do CEP, senao um casaria pedaco do outro.
#
# Todos exigem delimitador de palavra e separadores explicitos. Regex frouxa
# em texto clinico e perigosa no sentido oposto ao esperado: ela apagaria
# valores de exame e doses, destruindo a utilidade do dado.
# -----------------------------------------------------------------------------
PADROES: Final[tuple[tuple[TipoPII, Pattern[str]], ...]] = (
    (
        TipoPII.EMAIL,
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    ),
    (
        # Prontuario nos formatos "PRT-10001", "prontuario 10001", "pront. n 10001"
        TipoPII.PRONTUARIO,
        re.compile(
            r"\b(?:PRT-\d{4,}|pront(?:u[aá]rio)?\.?\s*(?:n[ºo°]?\.?\s*)?\d{4,})\b",
            re.IGNORECASE,
        ),
    ),
    (
        TipoPII.CRM,
        re.compile(r"\bCRM\s*[/-]?\s*[A-Z]{2}\s*[:\s-]?\s*\d{4,7}\b", re.IGNORECASE),
    ),
    (
        # Cartao Nacional de Saude: 15 digitos, normalmente em grupos 3-4-4-4.
        TipoPII.CNS,
        re.compile(r"\b\d{3}[\s.]\d{4}[\s.]\d{4}[\s.]\d{4}\b|\b\d{15}\b"),
    ),
    (
        # CPF exige a pontuacao OU 11 digitos corridos precedidos do rotulo.
        # Nao aceitamos 11 digitos soltos: em texto clinico isso poderia ser
        # um numero de registro de exame.
        TipoPII.CPF,
        re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|(?<=CPF )\s*\d{11}\b", re.IGNORECASE),
    ),
    (
        TipoPII.RG,
        re.compile(r"(?<=\bRG )\s*\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]\b|(?<=\bidentidade )\s*\d{1,2}\.?\d{3}\.?\d{3}-?[\dxX]\b", re.IGNORECASE),
    ),
    (
        # Telefone brasileiro: exige parenteses no DDD ou o hifen separador.
        TipoPII.TELEFONE,
        re.compile(r"\(\d{2}\)\s?9?\d{4}[-\s]?\d{4}\b|\b\d{2}\s9\d{4}-\d{4}\b"),
    ),
    (
        # CEP so e reconhecido com o hifen, para nao colidir com valores numericos.
        TipoPII.CEP,
        re.compile(r"\b\d{5}-\d{3}\b"),
    ),
    (
        # Data de nascimento: precisa do rotulo. Datas soltas em prontuario sao
        # datas de exame e de evolucao, que devem ser preservadas.
        TipoPII.DATA_NASCIMENTO,
        # Lookbehind sobre o rotulo: a data e substituida, mas "nascida em"
        # permanece no texto. O rotulo carrega sentido clinico e nao
        # identifica ninguem sozinho.
        re.compile(
            r"(?<=nascido em )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
            r"|(?<=nascida em )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
            r"|(?<=data de nascimento )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
            r"|(?<=data de nascimento: )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
            r"|(?<=\bDN )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
            r"|(?<=\bDN: )(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})",
            re.IGNORECASE,
        ),
    ),
    (
        # Idade acima de 89 anos e considerada identificavel em normas de
        # de-identificacao (o grupo e pequeno demais para diluir o individuo).
        TipoPII.IDADE_EXTREMA,
        re.compile(r"\b(?:9\d|1\d{2})\s*(?:anos|a\b)", re.IGNORECASE),
    ),
)

# -----------------------------------------------------------------------------
# NOMES DE PESSOA
# -----------------------------------------------------------------------------
# Nome proprio nao tem formato fixo, entao usamos duas estrategias
# complementares - e assumimos abertamente a limitacao de ambas:
#
#   1. DICIONARIO: nomes conhecidos (carregados do seed de pacientes e do
#      corpo clinico). Preciso, mas so pega quem esta na lista.
#   2. HEURISTICA DE ROTULO: sequencia de palavras capitalizadas logo apos um
#      marcador como "paciente", "Sr.", "Dra.". Pega nomes desconhecidos, mas
#      pode gerar falso positivo.
#
# Em producao hospitalar o correto seria um modelo de NER treinado em
# portugues clinico. Para o escopo deste projeto, com dados sinteticos cujos
# nomes conhecemos, as duas estrategias sao suficientes - e a limitacao esta
# documentada no relatorio tecnico.
# -----------------------------------------------------------------------------
MARCADORES_NOME: Final[tuple[str, ...]] = (
    "paciente", "sr", "sra", "senhor", "senhora", "dr", "dra", "doutor",
    "doutora", "enf", "enfermeiro", "enfermeira", "acompanhante", "responsavel",
)

# ATENÇÃO À AUSÊNCIA DE re.IGNORECASE — foi um defeito real deste projeto.
#
#   A primeira versão deste padrão usava re.IGNORECASE achando que isso só
#   tornava o MARCADOR insensível a caixa. Não é o que acontece: a flag vale
#   para a expressão inteira e anula as classes [A-Z] e [a-z], que são
#   justamente o que distingue um nome próprio de uma palavra comum.
#
#   O efeito foi grave e silencioso. Em "a avaliação do paciente deve incluir
#   a coleta", o trecho "paciente deve incluir" passava a casar como
#   "marcador + Nome Sobrenome", e o guardrail de saída reprovava toda
#   resposta do modelo por suposto vazamento de dado pessoal. O fluxo entrava
#   no ciclo de reescrita, esgotava as tentativas e degradava — em TODAS as
#   consultas.
#
#   A correção usa o grupo inline (?i:...) para tornar insensível a caixa
#   apenas a lista de marcadores, preservando a distinção de maiúsculas no
#   restante, que é o que identifica o nome próprio.
PADRAO_NOME_ROTULADO: Final[Pattern[str]] = re.compile(
    r"\b(?i:" + "|".join(MARCADORES_NOME) + r")\.?\s+"
    r"((?:[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-zàáâãéêíóôõúç]{2,}\s+)"
    r"(?:(?:d[aeo]s?|e)\s+)?"
    r"(?:[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-zàáâãéêíóôõúç]{2,}(?:\s+|$))+)",
    re.UNICODE,
)

# Palavras capitalizadas que NUNCA devem ser tratadas como nome de pessoa,
# mesmo que aparecam logo depois de um marcador.
NAO_SAO_NOMES: Final[frozenset[str]] = frozenset(
    {
        "hospital", "vida", "plena", "clinica", "medica", "pronto", "socorro",
        "uti", "unidade", "terapia", "intensiva", "enfermaria", "cardiologia",
        "nefrologia", "pneumologia", "endocrinologia", "neurologia",
        "gastroenterologia", "protocolo", "institucional", "comissao",
        "diabetes", "mellitus", "sepse", "choque", "septico", "avc",
        "pneumonia", "internado", "internada", "idoso", "idosa", "adulto",
        "masculino", "feminino", "anos", "leito", "setor",
    }
)


@dataclass(frozen=True)
class Achado:
    """
    Uma ocorrencia de PII localizada no texto.

    ATENCAO: o valor original NAO e armazenado. Guardamos apenas um hash
    truncado dele. Um relatorio de anonimizacao que registrasse os valores
    encontrados seria, ele proprio, um vazamento - o problema exato que este
    modulo existe para evitar.
    """

    tipo: TipoPII
    inicio: int
    fim: int
    tamanho: int
    hash_valor: str
    substituicao: str


class Anonimizador:
    """Localiza e substitui informacao identificavel em texto livre."""

    def __init__(
        self,
        politica: Politica = Politica.MASCARAR,
        *,
        nomes_conhecidos: Iterable[str] = (),
        chave: bytes | None = None,
    ) -> None:
        """
        Args:
            politica: MASCARAR (irreversivel, sem coerencia) ou PSEUDONIMIZAR
                (token estavel por valor, preserva coerencia referencial).
            nomes_conhecidos: nomes que devem ser substituidos sempre que
                aparecerem, mesmo sem marcador antes. Vem do seed de pacientes
                e da lista do corpo clinico.
            chave: segredo do HMAC usado na pseudonimizacao. Se omitida, e
                lida de MEDGRAPH_CHAVE_PSEUDONIMO ou sorteada. Sorteada
                significa que os tokens NAO sao estaveis entre execucoes - o
                que e seguro, porem impede comparar dois lotes processados em
                momentos diferentes. Para o dataset de treino, fixe a chave.
        """
        self.politica = politica
        self.chave = chave or os.getenv("MEDGRAPH_CHAVE_PSEUDONIMO", "").encode() or os.urandom(32)
        self.contagem: Counter[TipoPII] = Counter()
        # Mapa fragmento normalizado -> nome canonico. Garante que todas as
        # formas de citar a mesma pessoa produzam o mesmo pseudonimo.
        self._canonico: dict[str, str] = {}
        self._nomes = self._compilar_nomes(nomes_conhecidos)

    # -- nomes conhecidos ----------------------------------------------------
    @staticmethod
    def _normalizar(texto: str) -> str:
        """Minusculas e sem acento, para comparacao tolerante."""
        sem_acento = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()

    def _compilar_nomes(self, nomes: Iterable[str]) -> Pattern[str] | None:
        """
        Monta uma unica regex com todos os nomes conhecidos.

        Alem do nome completo, incluimos cada parte com 4 ou mais letras
        ("Maria Aparecida Souza" gera tambem "Aparecida" e "Souza"), porque
        evolucoes clinicas costumam citar o paciente pelo primeiro nome ou
        pelo sobrenome isolado. Partes curtas sao descartadas para nao
        transformar preposicoes em alvo.
        """
        alternativas: set[str] = set()
        for nome in nomes:
            nome = nome.strip()
            if len(nome) < 4:
                continue
            alternativas.add(re.escape(nome))
            self._canonico[self._normalizar(nome)] = nome
            for parte in nome.split():
                if len(parte) >= 4 and self._normalizar(parte) not in NAO_SAO_NOMES:
                    alternativas.add(re.escape(parte))
                    # Um sobrenome comum pode pertencer a mais de um paciente.
                    # Nesse caso o primeiro registrado vence: e melhor dois
                    # homonimos compartilharem um pseudonimo do que o mesmo
                    # paciente aparecer sob dois pseudonimos diferentes.
                    self._canonico.setdefault(self._normalizar(parte), nome)

        if not alternativas:
            return None

        # Ordena do maior para o menor: o nome completo precisa casar antes
        # de qualquer uma de suas partes.
        ordenadas = sorted(alternativas, key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(ordenadas) + r")\b", re.IGNORECASE)

    # -- substituicao --------------------------------------------------------
    def _token(self, tipo: TipoPII, valor: str) -> str:
        """Gera o texto que substitui o valor encontrado."""
        if self.politica is Politica.MASCARAR:
            return f"[{tipo.name}]"

        # Nomes sao resolvidos para a forma canonica antes do calculo, para
        # que "Maria Aparecida Souza", "Souza" e "Maria" gerem o MESMO token.
        base = valor.strip()
        if tipo is TipoPII.NOME:
            base = self._canonico.get(self._normalizar(base), base)

        # HMAC-SHA256 truncado: estavel para o mesmo par (valor, chave) e
        # inviavel de reverter sem a chave.
        digest = hmac.new(self.chave, base.lower().encode(), hashlib.sha256).hexdigest()
        return f"[{tipo.name}_{digest[:8]}]"

    @staticmethod
    def _hash(valor: str) -> str:
        """Hash curto do valor, apenas para contabilizar sem armazenar."""
        return hashlib.sha256(valor.encode()).hexdigest()[:12]

    # -- API principal -------------------------------------------------------
    def analisar(self, texto: str) -> list[Achado]:
        """
        Localiza PII SEM alterar o texto.

        Util para relatorios de auditoria do dataset: quantos identificadores
        de cada tipo existiam antes do tratamento.
        """
        achados: list[Achado] = []
        for tipo, padrao in PADROES:
            for m in padrao.finditer(texto):
                achados.append(
                    Achado(
                        tipo=tipo,
                        inicio=m.start(),
                        fim=m.end(),
                        tamanho=len(m.group()),
                        hash_valor=self._hash(m.group()),
                        substituicao=self._token(tipo, m.group()),
                    )
                )
        if self._nomes:
            for m in self._nomes.finditer(texto):
                achados.append(
                    Achado(
                        tipo=TipoPII.NOME,
                        inicio=m.start(),
                        fim=m.end(),
                        tamanho=len(m.group()),
                        hash_valor=self._hash(m.group()),
                        substituicao=self._token(TipoPII.NOME, m.group()),
                    )
                )

        # A heurística de nome rotulado também precisa entrar aqui: sem ela,
        # `analisar()` e `redigir()` dariam respostas diferentes sobre o mesmo
        # texto — e foi essa divergência que escondeu o defeito do IGNORACASE
        # durante a depuração, porque `analisar()` não encontrava nada e
        # `redigir()` encontrava.
        for m in PADRAO_NOME_ROTULADO.finditer(texto):
            candidato = m.group(1).strip()
            palavras = [p for p in candidato.split() if len(p) > 2]
            if not palavras or all(self._normalizar(p) in NAO_SAO_NOMES for p in palavras):
                continue
            achados.append(
                Achado(
                    tipo=TipoPII.NOME,
                    inicio=m.start(1),
                    fim=m.end(1),
                    tamanho=len(candidato),
                    hash_valor=self._hash(candidato),
                    substituicao=self._token(TipoPII.NOME, candidato),
                )
            )

        return sorted(achados, key=lambda a: a.inicio)

    def redigir(self, texto: str) -> str:
        """
        Devolve o texto com toda a PII substituida.

        Aplica os padroes na ordem definida em PADROES, depois o dicionario de
        nomes conhecidos e, por ultimo, a heuristica de nome rotulado.
        """
        if not texto:
            return texto

        resultado = texto

        for tipo, padrao in PADROES:
            def _sub(m: re.Match[str], _tipo: TipoPII = tipo) -> str:
                self.contagem[_tipo] += 1
                return self._token(_tipo, m.group())

            resultado = padrao.sub(_sub, resultado)

        if self._nomes:
            def _sub_nome(m: re.Match[str]) -> str:
                self.contagem[TipoPII.NOME] += 1
                return self._token(TipoPII.NOME, m.group())

            resultado = self._nomes.sub(_sub_nome, resultado)

        resultado = self._redigir_nomes_rotulados(resultado)
        return resultado

    def _redigir_nomes_rotulados(self, texto: str) -> str:
        """
        Heuristica para nomes que nao estao no dicionario.

        Substitui a sequencia capitalizada que vem logo depois de um marcador
        ("paciente Joao Pereira"), preservando o marcador. Desiste se todas as
        palavras estiverem na lista de termos que nunca sao nome - senao
        "paciente Clinica Medica" viraria "[NOME]".
        """

        def _sub(m: re.Match[str]) -> str:
            candidato = m.group(1).strip()
            palavras = [p for p in candidato.split() if len(p) > 2]
            if not palavras:
                return m.group(0)
            if all(self._normalizar(p) in NAO_SAO_NOMES for p in palavras):
                return m.group(0)
            if any(p.startswith("[") for p in palavras):  # ja anonimizado
                return m.group(0)

            self.contagem[TipoPII.NOME] += 1
            prefixo = m.group(0)[: m.start(1) - m.start(0)]
            return f"{prefixo}{self._token(TipoPII.NOME, candidato)} "

        return PADRAO_NOME_ROTULADO.sub(_sub, texto)

    # -- relatorio -----------------------------------------------------------
    def estatisticas(self) -> dict[str, int]:
        """Quantos identificadores de cada tipo foram substituidos ate agora."""
        return {tipo.value: qtd for tipo, qtd in sorted(self.contagem.items())}

    @property
    def total_removido(self) -> int:
        return sum(self.contagem.values())

    def registrar_na_auditoria(self, contexto: str) -> None:
        """Publica o resumo da anonimizacao na trilha de auditoria. [REQ-3b]"""
        registrar(
            TipoEvento.ANONIMIZACAO,
            f"{self.total_removido} identificador(es) removido(s) em {contexto}",
            contexto=contexto,
            politica=self.politica.value,
            por_tipo=self.estatisticas(),
        )


# -----------------------------------------------------------------------------
# INTEGRACAO COM A TRILHA DE AUDITORIA
# -----------------------------------------------------------------------------
_anonimizador_global: Anonimizador | None = None


def instalar_redator_de_auditoria(nomes_conhecidos: Iterable[str] = ()) -> Anonimizador:
    """
    [REQ-1a][REQ-3b] Faz a trilha de auditoria passar por este modulo.

    O modulo auditoria.py define um gancho `definir_redator` que e aplicado a
    todo texto antes de ir para o disco. Esta funcao instala o anonimizador
    nesse gancho. Deve ser chamada no bootstrap, antes da primeira consulta.

    Usa a politica MASCARAR: para log, saber que "havia um CPF ali" basta, e
    tokens estaveis num arquivo de auditoria permitiriam correlacionar
    registros de um mesmo paciente entre consultas - exatamente o tipo de
    reidentificacao que se quer impedir.
    """
    global _anonimizador_global
    from medgraph.auditoria import definir_redator

    _anonimizador_global = Anonimizador(
        politica=Politica.MASCARAR, nomes_conhecidos=nomes_conhecidos
    )
    definir_redator(_anonimizador_global.redigir)
    log.info(
        "Redator de PII instalado na trilha de auditoria (politica=mascarar, %d nome(s) conhecido(s))",
        len(list(nomes_conhecidos)),
    )
    return _anonimizador_global


def redigir(texto: str) -> str:
    """Atalho que usa o anonimizador global, criando um padrao se preciso."""
    global _anonimizador_global
    if _anonimizador_global is None:
        _anonimizador_global = Anonimizador(politica=Politica.MASCARAR)
    return _anonimizador_global.redigir(texto)
