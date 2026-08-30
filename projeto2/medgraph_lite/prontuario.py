"""
Base estruturada de pacientes.

O enunciado pede consulta a "base de dados estruturadas (como prontuarios e
registros)" e resposta "contextualizada com informacoes atualizadas do
paciente". Um SQLite de tres pacientes basta para demonstrar as duas coisas -
e cabe inteiro na tela durante a apresentacao.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

CAMINHO_PADRAO = "prontuarios.db"

# Tres pacientes, escolhidos para exercitar caminhos diferentes do fluxo:
# PAC-001 dispara conflito de alergia, PAC-002 dispara interacao medicamentosa,
# PAC-003 passa limpo.
PACIENTES = [
    {
        "id": "PAC-001", "idade": 72, "setor": "UTI",
        "alergias": "Penicilina (anafilaxia)",
        "medicacoes": "Noradrenalina, Omeprazol",
        "exames": "Lactato 4.5 mmol/L (critico); Creatinina 2.1 mg/dL (alterado)",
        "comorbidades": "Hipertensao, DPOC",
    },
    {
        "id": "PAC-002", "idade": 65, "setor": "Enfermaria",
        "alergias": "Nenhuma registrada",
        "medicacoes": "Varfarina, Metformina",
        "exames": "INR 2.8 (dentro da faixa); Hemoglobina 11.2 g/dL",
        "comorbidades": "Fibrilacao atrial, Diabetes tipo 2",
    },
    {
        "id": "PAC-003", "idade": 34, "setor": "Pronto-socorro",
        "alergias": "Nenhuma registrada",
        "medicacoes": "Nenhuma",
        "exames": "Hemograma normal; PCR 8 mg/L",
        "comorbidades": "Nenhuma",
    },
]


@dataclass
class Paciente:
    """Um paciente como o assistente o enxerga."""

    id: str
    idade: int
    setor: str
    alergias: str
    medicacoes: str
    exames: str
    comorbidades: str
    exames_criticos: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.exames_criticos = [
            parte.strip() for parte in self.exames.split(";") if "critico" in parte.lower()
        ]

    def resumo(self) -> str:
        """Bloco de texto que entra no prompt como contexto do paciente."""
        return (
            f"Paciente {self.id}, {self.idade} anos, {self.setor}.\n"
            f"Alergias: {self.alergias}\n"
            f"Medicacoes em uso: {self.medicacoes}\n"
            f"Exames: {self.exames}\n"
            f"Comorbidades: {self.comorbidades}"
        )


def criar_banco(caminho: str = CAMINHO_PADRAO) -> str:
    """Cria o banco do zero, de forma idempotente."""
    conexao = sqlite3.connect(caminho)
    conexao.execute("DROP TABLE IF EXISTS pacientes")
    conexao.execute(
        """CREATE TABLE pacientes (
               id TEXT PRIMARY KEY, idade INTEGER, setor TEXT, alergias TEXT,
               medicacoes TEXT, exames TEXT, comorbidades TEXT
           )"""
    )
    conexao.executemany(
        "INSERT INTO pacientes VALUES (:id, :idade, :setor, :alergias, "
        ":medicacoes, :exames, :comorbidades)",
        PACIENTES,
    )
    conexao.commit()
    conexao.close()
    return caminho


def buscar(paciente_id: str, caminho: str = CAMINHO_PADRAO) -> Paciente | None:
    """Devolve None quando o paciente nao existe - quem chama decide o que fazer."""
    conexao = sqlite3.connect(caminho)
    conexao.row_factory = sqlite3.Row
    linha = conexao.execute(
        "SELECT * FROM pacientes WHERE id = ?", (paciente_id,)
    ).fetchone()
    conexao.close()
    return Paciente(**dict(linha)) if linha else None


def listar(caminho: str = CAMINHO_PADRAO) -> list[str]:
    conexao = sqlite3.connect(caminho)
    ids = [linha[0] for linha in conexao.execute("SELECT id FROM pacientes")]
    conexao.close()
    return ids
