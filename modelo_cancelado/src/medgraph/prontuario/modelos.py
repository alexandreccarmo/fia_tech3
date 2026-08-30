"""
[REQ-2a] Modelos de domínio do prontuário.

O QUE FAZ:
    Define as estruturas que representam um paciente e seus registros
    clínicos, entre a consulta SQL e o resto do sistema.

POR QUE MODELOS TIPADOS E NÃO DICIONÁRIOS:
    O prontuário atravessa muitas fronteiras neste projeto: sai do SQLite,
    passa pelas regras clínicas, entra no prompt do modelo, aparece na trilha
    de auditoria e é exibido no painel. Com dicionários, cada uma dessas
    fronteiras teria que adivinhar quais chaves existem — e um `paciente["idade"]`
    onde a chave certa era `"idade_anos"` só falharia em tempo de execução,
    provavelmente no meio da demonstração.

    Com dataclasses, a estrutura é declarada uma vez e verificada pelo
    editor. Mais relevante ainda: propriedades como `alergico_a()` e
    `tem_valor_critico` ficam JUNTO do dado que interpretam, em vez de
    espalhadas como funções soltas.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any


def _normalizar(texto: str) -> str:
    """Minúsculas sem acento, para comparação tolerante de nomes de fármaco."""
    sem_acento = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower().strip()


@dataclass(frozen=True)
class Alergia:
    substancia: str
    classe: str | None = None
    gravidade: str | None = None
    reacao: str | None = None

    @property
    def e_grave(self) -> bool:
        return (self.gravidade or "").lower() == "grave"

    def corresponde_a(self, termo: str) -> bool:
        """
        Comparação TEXTUAL entre um termo e esta alergia.

        ATENÇÃO — ESTE MÉTODO NÃO DETECTA REATIVIDADE CRUZADA.

        Ele só reconhece o conflito quando o nome da substância ou o nome da
        classe aparece literalmente no termo. Um paciente com "Penicilina
        [Betalactâmico]" registrado e uma conduta citando "Ceftriaxona" NÃO
        produz correspondência aqui — as duas palavras não se parecem.

        Ligar ceftriaxona a betalactâmico exige conhecimento farmacológico,
        não similaridade de texto. Essa verificação, que é a que realmente
        importa, está em `medgraph.guardrails.regras_clinicas.verificar_alergias`,
        onde vive a tabela de classes de fármaco.

        Este método permanece por ser útil no caso simples — quando o próprio
        nome do fármaco é o alérgeno — mas nunca deve ser usado sozinho como
        checagem de segurança.
        """
        alvo = _normalizar(termo)
        if not alvo:
            return False
        return _normalizar(self.substancia) in alvo or (
            bool(self.classe) and _normalizar(self.classe) in alvo
        )


@dataclass(frozen=True)
class Medicacao:
    principio_ativo: str
    dose: str | None = None
    via: str | None = None
    frequencia: str | None = None
    inicio: str | None = None
    ativa: bool = True
    prescritor: str | None = None

    def __str__(self) -> str:
        partes = [self.principio_ativo]
        if self.dose:
            partes.append(self.dose)
        if self.via:
            partes.append(self.via)
        if self.frequencia:
            partes.append(self.frequencia)
        return " ".join(partes)


@dataclass(frozen=True)
class Exame:
    nome: str
    categoria: str | None = None
    solicitado_em: str | None = None
    resultado_em: str | None = None
    status: str = "pendente"
    valor: float | None = None
    unidade: str | None = None
    ref_min: float | None = None
    ref_max: float | None = None
    critico: bool = False
    laudo: str | None = None

    @property
    def pendente(self) -> bool:
        return self.status in ("pendente", "coletado")

    @property
    def fora_da_faixa(self) -> bool:
        """
        Se o resultado está fora da faixa de referência.

        Distinto de `critico`: alterado é estatístico, crítico é clínico.
        Potássio 5,2 mEq/L está alterado; 6,8 mEq/L é crítico e exige conduta
        imediata. Só o segundo dispara escalonamento no fluxo.
        """
        if self.valor is None:
            return False
        if self.ref_min is not None and self.valor < self.ref_min:
            return True
        return self.ref_max is not None and self.valor > self.ref_max

    @property
    def dias_pendente(self) -> int | None:
        """Há quantos dias o exame foi solicitado sem resultado."""
        if not self.pendente or not self.solicitado_em:
            return None
        try:
            return (date.today() - date.fromisoformat(self.solicitado_em)).days
        except ValueError:
            return None

    def __str__(self) -> str:
        if self.pendente:
            dias = self.dias_pendente
            sufixo = f" (pendente há {dias} dia{'s' if dias != 1 else ''})" if dias is not None else " (pendente)"
            return f"{self.nome}{sufixo}"
        valor = f"{self.valor}{' ' + self.unidade if self.unidade else ''}" if self.valor is not None else "—"
        faixa = (
            f" [ref {self.ref_min}–{self.ref_max}]"
            if self.ref_min is not None and self.ref_max is not None
            else ""
        )
        marca = " ** CRÍTICO **" if self.critico else (" (alterado)" if self.fora_da_faixa else "")
        return f"{self.nome}: {valor}{faixa}{marca}"


@dataclass(frozen=True)
class Comorbidade:
    descricao: str
    cid10: str | None = None
    desde: str | None = None


@dataclass(frozen=True)
class SinalVital:
    aferido_em: str
    pas: int | None = None
    pad: int | None = None
    fc: int | None = None
    fr: int | None = None
    temp: float | None = None
    sato2: int | None = None
    glasgow: int | None = None

    def __str__(self) -> str:
        partes = []
        if self.pas and self.pad:
            partes.append(f"PA {self.pas}/{self.pad} mmHg")
        if self.fc:
            partes.append(f"FC {self.fc} bpm")
        if self.fr:
            partes.append(f"FR {self.fr} irpm")
        if self.temp:
            partes.append(f"T {self.temp} °C")
        if self.sato2:
            partes.append(f"SatO₂ {self.sato2}%")
        if self.glasgow is not None:
            partes.append(f"Glasgow {self.glasgow}")
        return " · ".join(partes)


@dataclass(frozen=True)
class Evolucao:
    data: str
    texto: str
    autor: str | None = None
    especialidade: str | None = None


@dataclass
class Paciente:
    """Um paciente com todo o seu registro clínico carregado."""

    id: str
    prontuario: str
    nome: str
    data_nascimento: str
    sexo: str
    setor: str
    leito: str | None = None
    peso_kg: float | None = None
    altura_cm: float | None = None
    convenio: str | None = None
    data_internacao: str | None = None
    gestante: bool = False
    observacoes: str | None = None

    comorbidades: list[Comorbidade] = field(default_factory=list)
    alergias: list[Alergia] = field(default_factory=list)
    medicacoes: list[Medicacao] = field(default_factory=list)
    exames: list[Exame] = field(default_factory=list)
    sinais_vitais: list[SinalVital] = field(default_factory=list)
    evolucoes: list[Evolucao] = field(default_factory=list)

    # -- derivados -----------------------------------------------------------
    @property
    def idade(self) -> int:
        nascimento = date.fromisoformat(self.data_nascimento)
        hoje = date.today()
        return hoje.year - nascimento.year - (
            (hoje.month, hoje.day) < (nascimento.month, nascimento.day)
        )

    @property
    def pediatrico(self) -> bool:
        return self.idade < 18

    @property
    def idoso(self) -> bool:
        return self.idade >= 65

    @property
    def populacao_especial(self) -> bool:
        """
        Populações que exigem ajuste de dose e têm contraindicações próprias.

        Gestantes, pediátricos e idosos acima de 80 anos. O fluxo eleva o
        escore de risco quando o paciente se enquadra aqui.
        """
        return self.gestante or self.pediatrico or self.idade >= 80

    @property
    def medicacoes_ativas(self) -> list[Medicacao]:
        return [m for m in self.medicacoes if m.ativa]

    @property
    def exames_pendentes(self) -> list[Exame]:
        return [e for e in self.exames if e.pendente]

    @property
    def exames_criticos(self) -> list[Exame]:
        return [e for e in self.exames if e.critico]

    @property
    def tem_valor_critico(self) -> bool:
        return bool(self.exames_criticos)

    @property
    def ultimo_sinal_vital(self) -> SinalVital | None:
        if not self.sinais_vitais:
            return None
        return max(self.sinais_vitais, key=lambda s: s.aferido_em)

    def alergico_a(self, termo: str) -> list[Alergia]:
        """
        Alergias cujo nome aparece literalmente no termo.

        Verificação TEXTUAL apenas. Para a checagem de segurança de verdade,
        que considera reatividade cruzada entre classes de fármaco, use
        `medgraph.guardrails.regras_clinicas.verificar_alergias`. Ver a nota
        em `Alergia.corresponde_a`.
        """
        return [a for a in self.alergias if a.corresponde_a(termo)]

    def usa_medicacao(self, principio: str) -> list[Medicacao]:
        alvo = _normalizar(principio)
        return [m for m in self.medicacoes_ativas if alvo in _normalizar(m.principio_ativo)]

    # -- apresentação --------------------------------------------------------
    def resumo_clinico(self, *, anonimo: bool = True) -> str:
        """
        Bloco de contexto do paciente injetado no prompt do modelo.  [REQ-2b]

        POR QUE `anonimo=True` É O PADRÃO:
            Este texto vai para dentro do prompt do modelo e, de lá, para a
            trilha de auditoria. O nome do paciente não acrescenta nada ao
            raciocínio clínico — idade, sexo, comorbidades, alergias e
            medicações, sim. Omitir o identificador por padrão significa que
            o caminho preguiçoso é também o caminho seguro.

            O prontuário é sempre referenciado pelo id interno, que permite
            ao médico localizar o paciente no sistema do hospital sem que o
            nome transite pelo modelo.
        """
        linhas: list[str] = []
        identificacao = f"Paciente {self.id}" if anonimo else f"{self.nome} ({self.id})"
        linhas.append(
            f"{identificacao} · {self.idade} anos · sexo {self.sexo} · {self.setor}"
            + (f" · leito {self.leito}" if self.leito else "")
        )

        marcadores: list[str] = []
        if self.gestante:
            marcadores.append("GESTANTE")
        if self.pediatrico:
            marcadores.append("PEDIÁTRICO")
        if self.idade >= 80:
            marcadores.append("IDOSO ≥ 80 ANOS")
        if marcadores:
            linhas.append("População especial: " + ", ".join(marcadores))

        if self.comorbidades:
            linhas.append(
                "Comorbidades: "
                + "; ".join(
                    f"{c.descricao}" + (f" ({c.cid10})" if c.cid10 else "")
                    for c in self.comorbidades
                )
            )

        if self.alergias:
            linhas.append(
                "ALERGIAS: "
                + "; ".join(
                    f"{a.substancia}"
                    + (f" [{a.classe}]" if a.classe else "")
                    + (f" — {a.gravidade}" if a.gravidade else "")
                    + (f", {a.reacao}" if a.reacao else "")
                    for a in self.alergias
                )
            )
        else:
            linhas.append("ALERGIAS: nenhuma registrada")

        if self.medicacoes_ativas:
            linhas.append(
                "Medicações em uso: " + "; ".join(str(m) for m in self.medicacoes_ativas)
            )

        criticos = self.exames_criticos
        if criticos:
            linhas.append("EXAMES EM VALOR CRÍTICO: " + "; ".join(str(e) for e in criticos))

        alterados = [e for e in self.exames if e.fora_da_faixa and not e.critico]
        if alterados:
            linhas.append("Exames alterados: " + "; ".join(str(e) for e in alterados[:6]))

        pendentes = self.exames_pendentes
        if pendentes:
            linhas.append("Exames pendentes: " + "; ".join(str(e) for e in pendentes))

        vital = self.ultimo_sinal_vital
        if vital:
            linhas.append(f"Últimos sinais vitais ({vital.aferido_em}): {vital}")

        if self.evolucoes:
            recente = max(self.evolucoes, key=lambda e: e.data)
            linhas.append(f"Última evolução ({recente.data}): {recente.texto}")

        return "\n".join(linhas)

    def para_dict(self, *, anonimo: bool = True) -> dict[str, Any]:
        """Representação serializável, para o painel e para a auditoria."""
        return {
            "id": self.id,
            "prontuario": self.prontuario if not anonimo else "(omitido)",
            "nome": self.nome if not anonimo else "(omitido)",
            "idade": self.idade,
            "sexo": self.sexo,
            "setor": self.setor,
            "leito": self.leito,
            "gestante": self.gestante,
            "populacao_especial": self.populacao_especial,
            "comorbidades": [c.descricao for c in self.comorbidades],
            "alergias": [
                {"substancia": a.substancia, "classe": a.classe, "gravidade": a.gravidade}
                for a in self.alergias
            ],
            "medicacoes_ativas": [str(m) for m in self.medicacoes_ativas],
            "exames_pendentes": [str(e) for e in self.exames_pendentes],
            "exames_criticos": [str(e) for e in self.exames_criticos],
            "ultimo_sinal_vital": str(self.ultimo_sinal_vital) if self.ultimo_sinal_vital else None,
        }
