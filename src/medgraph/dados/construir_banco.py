"""
[REQ-2a][REQ-E2] Construcao da base estruturada de prontuarios.

O QUE FAZ:
    Le data/sintetico/pacientes_seed.json e monta um banco SQLite normalizado
    com sete tabelas: pacientes, comorbidades, alergias, medicacoes, exames,
    sinais vitais e evolucoes.

POR QUE UM BANCO RELACIONAL, E NAO O PROPRIO JSON:
    O enunciado pede, no item 2, "realizar consultas em base de dados
    estruturadas (como prontuarios e registros)". Ler um JSON e carrega-lo
    em memoria nao e consultar uma base estruturada - e ler um arquivo.

    A diferenca importa na pratica. O assistente precisa responder perguntas
    como "quais exames deste paciente estao pendentes ha mais de 48 horas?"
    ou "esta conduta conflita com alguma alergia registrada?". Em SQL isso e
    uma consulta com indice; em JSON e um laco que percorre tudo. Alem disso,
    o SQLite modela explicitamente as CHAVES ESTRANGEIRAS entre paciente e
    seus registros clinicos, o que impede estados inconsistentes.

POR QUE OS DADOS PESSOAIS FICAM NO BANCO:
    Nome, CPF e telefone sao gravados como vieram (sinteticos). A anonimizacao
    NAO acontece aqui, e sim na fronteira de saida: o guardrail verifica que
    nenhum identificador vazou para a resposta entregue ao medico. Esse e o
    desenho correto - um prontuario sem nome do paciente seria inutil para o
    hospital; o que nao pode e o identificador escapar para o modelo, para a
    trilha de auditoria ou para o texto final.

Uso:
    python -m medgraph.dados.construir_banco
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

# -----------------------------------------------------------------------------
# ESQUEMA
# -----------------------------------------------------------------------------
# Notas de projeto:
#   - Todas as tabelas filhas usam ON DELETE CASCADE: remover um paciente leva
#     junto todo o seu registro clinico, sem deixar orfaos.
#   - Os campos de faixa de referencia (ref_min / ref_max) ficam junto do
#     resultado, e nao numa tabela de exames-catalogo. Faixas de referencia
#     mudam com o metodo do laboratorio, entao guarda-las com o resultado
#     preserva a interpretacao correta na epoca da coleta.
#   - `critico` e materializado como coluna em vez de ser calculado na hora.
#     A criticidade de um valor depende de julgamento clinico, nao apenas de
#     estar fora da faixa - potassio 5,2 esta alterado, potassio 6,8 e critico.
# -----------------------------------------------------------------------------
ESQUEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pacientes (
    id                TEXT PRIMARY KEY,
    prontuario        TEXT NOT NULL UNIQUE,
    nome              TEXT NOT NULL,
    cpf               TEXT,
    data_nascimento   TEXT NOT NULL,
    sexo              TEXT NOT NULL CHECK (sexo IN ('M','F')),
    peso_kg           REAL,
    altura_cm         REAL,
    convenio          TEXT,
    setor             TEXT NOT NULL,
    leito             TEXT,
    data_internacao   TEXT,
    gestante          INTEGER NOT NULL DEFAULT 0 CHECK (gestante IN (0,1)),
    telefone          TEXT,
    email             TEXT,
    observacoes       TEXT
);

CREATE TABLE IF NOT EXISTS comorbidades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    cid10       TEXT,
    descricao   TEXT NOT NULL,
    desde       TEXT
);

CREATE TABLE IF NOT EXISTS alergias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    substancia  TEXT NOT NULL,
    classe      TEXT,
    gravidade   TEXT CHECK (gravidade IN ('leve','moderada','grave')),
    reacao      TEXT
);

CREATE TABLE IF NOT EXISTS medicacoes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id     TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    principio_ativo TEXT NOT NULL,
    dose            TEXT,
    via             TEXT CHECK (via IN ('VO','EV','IM','SC','IN','TOP','SL')),
    frequencia      TEXT,
    inicio          TEXT,
    ativa           INTEGER NOT NULL DEFAULT 1 CHECK (ativa IN (0,1)),
    prescritor      TEXT
);

CREATE TABLE IF NOT EXISTS exames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id   TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    nome          TEXT NOT NULL,
    categoria     TEXT,
    solicitado_em TEXT,
    resultado_em  TEXT,
    status        TEXT NOT NULL CHECK (status IN ('resultado','pendente','coletado')),
    valor         REAL,
    unidade       TEXT,
    ref_min       REAL,
    ref_max       REAL,
    critico       INTEGER NOT NULL DEFAULT 0 CHECK (critico IN (0,1)),
    laudo         TEXT
);

CREATE TABLE IF NOT EXISTS sinais_vitais (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    aferido_em  TEXT NOT NULL,
    pas         INTEGER,
    pad         INTEGER,
    fc          INTEGER,
    fr          INTEGER,
    temp        REAL,
    sato2       INTEGER,
    glasgow     INTEGER
);

CREATE TABLE IF NOT EXISTS evolucoes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id   TEXT NOT NULL REFERENCES pacientes(id) ON DELETE CASCADE,
    data          TEXT NOT NULL,
    autor         TEXT,
    especialidade TEXT,
    texto         TEXT NOT NULL
);

-- Indices nas consultas que o assistente realmente faz:
-- "exames pendentes deste paciente", "alergias deste paciente",
-- "medicacoes ativas deste paciente".
CREATE INDEX IF NOT EXISTS idx_exames_paciente_status ON exames(paciente_id, status);
CREATE INDEX IF NOT EXISTS idx_exames_critico        ON exames(paciente_id, critico);
CREATE INDEX IF NOT EXISTS idx_alergias_paciente     ON alergias(paciente_id);
CREATE INDEX IF NOT EXISTS idx_medicacoes_ativas     ON medicacoes(paciente_id, ativa);
CREATE INDEX IF NOT EXISTS idx_evolucoes_paciente    ON evolucoes(paciente_id, data);
CREATE INDEX IF NOT EXISTS idx_vitais_paciente       ON sinais_vitais(paciente_id, aferido_em);
CREATE INDEX IF NOT EXISTS idx_pacientes_prontuario  ON pacientes(prontuario);

-- Visao de conveniencia: o "cartao de identificacao clinica" do paciente,
-- que e o primeiro bloco de contexto injetado no prompt do assistente.
CREATE VIEW IF NOT EXISTS vw_resumo_paciente AS
SELECT
    p.id,
    p.prontuario,
    p.nome,
    p.sexo,
    p.setor,
    p.leito,
    p.gestante,
    CAST((julianday('now') - julianday(p.data_nascimento)) / 365.25 AS INTEGER) AS idade,
    (SELECT COUNT(*) FROM alergias    a WHERE a.paciente_id = p.id)                        AS qtd_alergias,
    (SELECT COUNT(*) FROM medicacoes  m WHERE m.paciente_id = p.id AND m.ativa = 1)        AS qtd_medicacoes_ativas,
    (SELECT COUNT(*) FROM exames      e WHERE e.paciente_id = p.id AND e.status='pendente')AS qtd_exames_pendentes,
    (SELECT COUNT(*) FROM exames      e WHERE e.paciente_id = p.id AND e.critico = 1)      AS qtd_exames_criticos
FROM pacientes p;
"""


def _idade(data_nascimento: str) -> int:
    nascimento = date.fromisoformat(data_nascimento)
    hoje = date.today()
    return hoje.year - nascimento.year - (
        (hoje.month, hoje.day) < (nascimento.month, nascimento.day)
    )


def construir(cfg: Settings | None = None, *, forcar: bool = False) -> dict[str, int]:
    """
    Cria o banco SQLite a partir do seed sintetico.

    Args:
        forcar: recria o banco do zero mesmo que ja exista.

    Returns:
        Quantidade de linhas inseridas por tabela.
    """
    cfg = cfg or obter_settings()
    caminho_seed = cfg.dir_dados_sinteticos / "pacientes_seed.json"
    destino = cfg.caminho_banco_prontuarios

    if not caminho_seed.exists():
        raise FileNotFoundError(
            f"{caminho_seed} nao encontrado. O seed sintetico faz parte do repositorio."
        )

    if destino.exists():
        if not forcar:
            log.info("%s ja existe - use forcar=True para recriar.", destino.name)
            return _contar_linhas(destino)
        destino.unlink()

    seed = json.loads(caminho_seed.read_text(encoding="utf-8"))
    pacientes: list[dict[str, Any]] = seed["pacientes"]

    with etapa("construir_banco", tipo=TipoEvento.BANCO, pacientes=len(pacientes)):
        destino.parent.mkdir(parents=True, exist_ok=True)
        conexao = sqlite3.connect(destino)
        try:
            conexao.executescript(ESQUEMA)
            contagem: dict[str, int] = dict.fromkeys(
                ("pacientes", "comorbidades", "alergias", "medicacoes",
                 "exames", "sinais_vitais", "evolucoes"),
                0,
            )

            for p in pacientes:
                conexao.execute(
                    """INSERT INTO pacientes (id, prontuario, nome, cpf, data_nascimento, sexo,
                                              peso_kg, altura_cm, convenio, setor, leito,
                                              data_internacao, gestante, telefone, email, observacoes)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        p["id"], p["prontuario"], p["nome"], p.get("cpf"),
                        p["data_nascimento"], p["sexo"], p.get("peso_kg"), p.get("altura_cm"),
                        p.get("convenio"), p["setor"], p.get("leito"), p.get("data_internacao"),
                        int(bool(p.get("gestante"))), p.get("telefone"), p.get("email"),
                        p.get("observacoes"),
                    ),
                )
                contagem["pacientes"] += 1

                for c in p.get("comorbidades", []):
                    conexao.execute(
                        "INSERT INTO comorbidades (paciente_id, cid10, descricao, desde) VALUES (?,?,?,?)",
                        (p["id"], c.get("cid10"), c["descricao"], c.get("desde")),
                    )
                    contagem["comorbidades"] += 1

                for a in p.get("alergias", []):
                    conexao.execute(
                        "INSERT INTO alergias (paciente_id, substancia, classe, gravidade, reacao) VALUES (?,?,?,?,?)",
                        (p["id"], a["substancia"], a.get("classe"), a.get("gravidade"), a.get("reacao")),
                    )
                    contagem["alergias"] += 1

                for m in p.get("medicacoes", []):
                    conexao.execute(
                        """INSERT INTO medicacoes (paciente_id, principio_ativo, dose, via,
                                                   frequencia, inicio, ativa, prescritor)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            p["id"], m["principio_ativo"], m.get("dose"), m.get("via"),
                            m.get("frequencia"), m.get("inicio"), int(bool(m.get("ativa", True))),
                            m.get("prescritor"),
                        ),
                    )
                    contagem["medicacoes"] += 1

                for e in p.get("exames", []):
                    conexao.execute(
                        """INSERT INTO exames (paciente_id, nome, categoria, solicitado_em,
                                               resultado_em, status, valor, unidade,
                                               ref_min, ref_max, critico, laudo)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            p["id"], e["nome"], e.get("categoria"), e.get("solicitado_em"),
                            e.get("resultado_em"), e["status"], e.get("valor"), e.get("unidade"),
                            e.get("ref_min"), e.get("ref_max"), int(bool(e.get("critico"))),
                            e.get("laudo"),
                        ),
                    )
                    contagem["exames"] += 1

                for v in p.get("sinais_vitais", []):
                    conexao.execute(
                        """INSERT INTO sinais_vitais (paciente_id, aferido_em, pas, pad, fc, fr,
                                                      temp, sato2, glasgow)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            p["id"], v["aferido_em"], v.get("pas"), v.get("pad"), v.get("fc"),
                            v.get("fr"), v.get("temp"), v.get("sato2"), v.get("glasgow"),
                        ),
                    )
                    contagem["sinais_vitais"] += 1

                for ev in p.get("evolucoes", []):
                    conexao.execute(
                        "INSERT INTO evolucoes (paciente_id, data, autor, especialidade, texto) VALUES (?,?,?,?,?)",
                        (p["id"], ev["data"], ev.get("autor"), ev.get("especialidade"), ev["texto"]),
                    )
                    contagem["evolucoes"] += 1

            conexao.commit()
        finally:
            conexao.close()

    log.info("Banco criado em %s", destino)
    for tabela, qtd in contagem.items():
        log.info("  %-16s %5d linhas", tabela, qtd)

    registrar(
        TipoEvento.BANCO,
        "Base de prontuarios construida",
        etapa="construir_banco",
        conclusao=True,
        caminho=str(destino.name),
        **contagem,
    )
    return contagem


def _contar_linhas(caminho) -> dict[str, int]:
    conexao = sqlite3.connect(caminho)
    try:
        tabelas = [
            "pacientes", "comorbidades", "alergias", "medicacoes",
            "exames", "sinais_vitais", "evolucoes",
        ]
        return {t: conexao.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tabelas}
    finally:
        conexao.close()


def estatisticas(cfg: Settings | None = None) -> dict[str, Any]:
    """
    Numeros que caracterizam a base. Vao para o relatorio tecnico e para o
    painel visual, e servem de verificacao rapida de que o seed foi carregado
    com os casos que o sistema precisa exercitar.
    """
    cfg = cfg or obter_settings()
    conexao = sqlite3.connect(cfg.caminho_banco_prontuarios)
    conexao.row_factory = sqlite3.Row
    try:
        def um(sql: str) -> Any:
            return conexao.execute(sql).fetchone()[0]

        return {
            "pacientes": um("SELECT COUNT(*) FROM pacientes"),
            "idade_media": round(
                sum(_idade(r[0]) for r in conexao.execute("SELECT data_nascimento FROM pacientes"))
                / max(1, um("SELECT COUNT(*) FROM pacientes")),
                1,
            ),
            "gestantes": um("SELECT COUNT(*) FROM pacientes WHERE gestante = 1"),
            "com_alergia": um("SELECT COUNT(DISTINCT paciente_id) FROM alergias"),
            "com_alergia_penicilina": um(
                "SELECT COUNT(DISTINCT paciente_id) FROM alergias "
                "WHERE substancia LIKE '%enicilina%' OR classe LIKE '%etalact%'"
            ),
            "com_exame_pendente": um(
                "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE status = 'pendente'"
            ),
            "com_exame_critico": um(
                "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE critico = 1"
            ),
            "medicacoes_ativas": um("SELECT COUNT(*) FROM medicacoes WHERE ativa = 1"),
            "setores": [
                dict(r) for r in conexao.execute(
                    "SELECT setor, COUNT(*) AS total FROM pacientes GROUP BY setor ORDER BY total DESC"
                )
            ],
        }
    finally:
        conexao.close()


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Base de prontuarios", subtitulo="SQLite a partir do seed sintetico")
    construir(forcar=True)
    print(json.dumps(estatisticas(), ensure_ascii=False, indent=2))
