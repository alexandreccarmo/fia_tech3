"""
[REQ-2a][REQ-2b] Consulta à base estruturada de prontuários.

O QUE FAZ:
    Traduz as perguntas que o assistente precisa fazer sobre um paciente em
    consultas SQL, e devolve objetos de domínio prontos para uso.

POR QUE UM REPOSITÓRIO, E NÃO SQL ESPALHADO PELOS NÓS DO GRAFO:
    O item 2 do enunciado pede "realizar consultas em base de dados
    estruturadas". A tentação é escrever a consulta onde ela é usada. Três
    razões para não fazer isso:

    1. AUDITORIA. Toda consulta a dado de paciente é registrada na trilha,
       com o identificador do paciente e o tipo de acesso. Concentrando o
       acesso aqui, esse registro é automático; espalhado, dependeria de
       alguém lembrar em cada ponto.
    2. SUPERFÍCIE DE EXPOSIÇÃO. Existe exatamente um lugar no projeto que
       lê dados de paciente. Auditar quem acessa o quê é ler um arquivo.
    3. PROTEÇÃO CONTRA ACESSO EM MASSA. O repositório não expõe nenhum método
       que devolva todos os pacientes com seus dados clínicos. É uma decisão
       de segurança, não um esquecimento — ver `listar_pacientes`.

CONEXÃO SOMENTE LEITURA:
    O assistente NUNCA escreve no prontuário. A conexão é aberta em modo
    read-only pela própria URI do SQLite, então uma tentativa de escrita
    falha no driver, e não numa convenção que alguém pode contornar. Isso é
    parte dos limites de atuação exigidos pelo item 3: o sistema pode ler o
    prontuário para contextualizar, jamais alterá-lo.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, registrar
from medgraph.logging_config import obter_logger
from medgraph.prontuario.modelos import (
    Alergia,
    Comorbidade,
    Evolucao,
    Exame,
    Medicacao,
    Paciente,
    SinalVital,
)

log = obter_logger(__name__)

# Teto de registros devolvidos em qualquer consulta que aceite filtro livre.
# Existe para que um erro de programação, ou um pedido malicioso que
# escapasse do guardrail, não consiga extrair a base inteira numa chamada.
LIMITE_RESULTADOS = 25


class PacienteNaoEncontradoError(LookupError):
    """O identificador informado não corresponde a nenhum paciente."""


class RepositorioProntuarios:
    """Acesso somente leitura à base de prontuários."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or obter_settings()
        self.caminho: Path = self.cfg.caminho_banco_prontuarios
        if not self.caminho.exists():
            raise FileNotFoundError(
                f"Base de prontuários não encontrada em {self.caminho}.\n"
                f"Para criar:  make dados"
            )

    @contextmanager
    def _conexao(self) -> Iterator[sqlite3.Connection]:
        """
        Abre uma conexão SOMENTE LEITURA.

        O `mode=ro` na URI é o que garante a restrição no nível do driver:
        qualquer INSERT, UPDATE ou DELETE levanta sqlite3.OperationalError.
        Uma convenção de código ("não escreva aqui") seria só um pedido.
        """
        conexao = sqlite3.connect(f"file:{self.caminho}?mode=ro", uri=True)
        conexao.row_factory = sqlite3.Row
        try:
            yield conexao
        finally:
            conexao.close()

    # =========================================================================
    # CARGA COMPLETA DE UM PACIENTE
    # =========================================================================
    def obter_paciente(self, identificador: str) -> Paciente:
        """
        Carrega um paciente com todo o seu registro clínico.  [REQ-2b]

        Aceita o id interno (PAC-0001) ou o número de prontuário (PRT-10001) —
        na prática o médico usa um ou outro conforme o sistema de origem.

        Uma única chamada traz tudo. Poderia ser preguiçoso, carregando exames
        e medicações só quando acessados, mas o assistente precisa do quadro
        completo para raciocinar: uma conduta segura depende de conhecer as
        alergias E as medicações E os exames críticos ao mesmo tempo. Carga
        parcial abriria a possibilidade de o modelo decidir sem ver uma
        contraindicação que estava no banco.
        """
        with self._conexao() as conexao:
            linha = conexao.execute(
                "SELECT * FROM pacientes WHERE id = ? OR prontuario = ?",
                (identificador, identificador),
            ).fetchone()

            if linha is None:
                registrar(
                    TipoEvento.BANCO,
                    "Consulta a paciente inexistente",
                    nivel="WARNING",
                    identificador=identificador,
                )
                raise PacienteNaoEncontradoError(
                    f"Nenhum paciente com id ou prontuário '{identificador}'."
                )

            paciente_id = linha["id"]
            paciente = Paciente(
                id=linha["id"],
                prontuario=linha["prontuario"],
                nome=linha["nome"],
                data_nascimento=linha["data_nascimento"],
                sexo=linha["sexo"],
                setor=linha["setor"],
                leito=linha["leito"],
                peso_kg=linha["peso_kg"],
                altura_cm=linha["altura_cm"],
                convenio=linha["convenio"],
                data_internacao=linha["data_internacao"],
                gestante=bool(linha["gestante"]),
                observacoes=linha["observacoes"],
                comorbidades=[
                    Comorbidade(descricao=r["descricao"], cid10=r["cid10"], desde=r["desde"])
                    for r in conexao.execute(
                        "SELECT * FROM comorbidades WHERE paciente_id = ?", (paciente_id,)
                    )
                ],
                alergias=[
                    Alergia(
                        substancia=r["substancia"], classe=r["classe"],
                        gravidade=r["gravidade"], reacao=r["reacao"],
                    )
                    for r in conexao.execute(
                        "SELECT * FROM alergias WHERE paciente_id = ?", (paciente_id,)
                    )
                ],
                medicacoes=[
                    Medicacao(
                        principio_ativo=r["principio_ativo"], dose=r["dose"], via=r["via"],
                        frequencia=r["frequencia"], inicio=r["inicio"],
                        ativa=bool(r["ativa"]), prescritor=r["prescritor"],
                    )
                    for r in conexao.execute(
                        "SELECT * FROM medicacoes WHERE paciente_id = ? ORDER BY ativa DESC, principio_ativo",
                        (paciente_id,),
                    )
                ],
                exames=[
                    Exame(
                        nome=r["nome"], categoria=r["categoria"],
                        solicitado_em=r["solicitado_em"], resultado_em=r["resultado_em"],
                        status=r["status"], valor=r["valor"], unidade=r["unidade"],
                        ref_min=r["ref_min"], ref_max=r["ref_max"],
                        critico=bool(r["critico"]), laudo=r["laudo"],
                    )
                    for r in conexao.execute(
                        "SELECT * FROM exames WHERE paciente_id = ? "
                        "ORDER BY critico DESC, solicitado_em DESC",
                        (paciente_id,),
                    )
                ],
                sinais_vitais=[
                    SinalVital(
                        aferido_em=r["aferido_em"], pas=r["pas"], pad=r["pad"], fc=r["fc"],
                        fr=r["fr"], temp=r["temp"], sato2=r["sato2"], glasgow=r["glasgow"],
                    )
                    for r in conexao.execute(
                        "SELECT * FROM sinais_vitais WHERE paciente_id = ? ORDER BY aferido_em DESC",
                        (paciente_id,),
                    )
                ],
                evolucoes=[
                    Evolucao(
                        data=r["data"], texto=r["texto"],
                        autor=r["autor"], especialidade=r["especialidade"],
                    )
                    for r in conexao.execute(
                        "SELECT * FROM evolucoes WHERE paciente_id = ? ORDER BY data DESC",
                        (paciente_id,),
                    )
                ],
            )

        registrar(
            TipoEvento.BANCO,
            f"Prontuário de {paciente.id} consultado",
            paciente_id=paciente.id,
            setor=paciente.setor,
            idade=paciente.idade,
            alergias=len(paciente.alergias),
            medicacoes_ativas=len(paciente.medicacoes_ativas),
            exames_pendentes=len(paciente.exames_pendentes),
            exames_criticos=len(paciente.exames_criticos),
        )
        return paciente

    # =========================================================================
    # CONSULTAS ESPECÍFICAS
    # =========================================================================
    def exames_pendentes(self, identificador: str) -> list[Exame]:
        """
        Exames solicitados e ainda sem resultado.

        É uma das cinco intenções que o assistente reconhece, e o enunciado a
        cita explicitamente como exemplo de etapa do fluxo automatizado.
        """
        paciente = self.obter_paciente(identificador)
        pendentes = paciente.exames_pendentes
        registrar(
            TipoEvento.BANCO,
            f"{len(pendentes)} exame(s) pendente(s) para {paciente.id}",
            paciente_id=paciente.id,
            exames=[e.nome for e in pendentes],
        )
        return pendentes

    def valores_criticos(self, identificador: str) -> list[Exame]:
        """Resultados em faixa crítica — disparam escalonamento no fluxo."""
        return self.obter_paciente(identificador).exames_criticos

    def verificar_alergia(self, identificador: str, termo: str) -> list[Alergia]:
        """
        Alergias que colidem com um fármaco ou classe citada.  [REQ-3a]

        Base da regra de segurança mais importante do sistema. A comparação
        considera a CLASSE além da substância: um paciente alérgico a
        penicilina (classe betalactâmico) não pode receber ceftriaxona, e o
        texto da conduta jamais vai citar a palavra "penicilina".
        """
        paciente = self.obter_paciente(identificador)
        achados = paciente.alergico_a(termo)
        if achados:
            registrar(
                TipoEvento.REGRA_CLINICA,
                f"Alergia registrada colide com '{termo}'",
                nivel="WARNING",
                paciente_id=paciente.id,
                termo=termo,
                alergias=[a.substancia for a in achados],
                gravidade=[a.gravidade for a in achados],
            )
        return achados

    # =========================================================================
    # LISTAGEM — deliberadamente limitada
    # =========================================================================
    def listar_pacientes(self, *, setor: str | None = None, limite: int = LIMITE_RESULTADOS) -> list[dict[str, Any]]:
        """
        Lista pacientes com dados MÍNIMOS de identificação assistencial.

        POR QUE ESTE MÉTODO NÃO DEVOLVE O REGISTRO CLÍNICO:
            Serve para popular o seletor de paciente no painel — e nada além
            disso. Devolve id, setor, leito, idade e as contagens que aparecem
            na lista; não devolve nome, diagnóstico, medicações nem resultados
            de exame.

            Um método que retornasse "todos os pacientes com seus dados" seria
            a maneira mais fácil de esvaziar a base numa chamada só, e
            existiria justamente para o pedido que o guardrail de entrada
            bloqueia ("liste todos os pacientes com..."). Não construí-lo é
            mais eficaz do que construí-lo e proteger depois.

            O limite padrão de 25 é um segundo freio, para o caso de este
            método ser chamado de um ponto que não esperávamos.
        """
        limite = min(limite, LIMITE_RESULTADOS)
        with self._conexao() as conexao:
            if setor:
                linhas = conexao.execute(
                    "SELECT * FROM vw_resumo_paciente WHERE setor = ? ORDER BY id LIMIT ?",
                    (setor, limite),
                ).fetchall()
            else:
                linhas = conexao.execute(
                    "SELECT * FROM vw_resumo_paciente ORDER BY id LIMIT ?", (limite,)
                ).fetchall()

        registrar(
            TipoEvento.BANCO,
            f"Listagem de {len(linhas)} paciente(s)",
            setor=setor or "(todos)",
            limite=limite,
        )
        return [dict(linha) for linha in linhas]

    def setores(self) -> list[str]:
        with self._conexao() as conexao:
            return [
                r[0] for r in conexao.execute(
                    "SELECT DISTINCT setor FROM pacientes ORDER BY setor"
                )
            ]

    def estatisticas(self) -> dict[str, Any]:
        """Números agregados da base. Não expõem nenhum paciente individual."""
        with self._conexao() as conexao:
            def um(sql: str) -> int:
                return conexao.execute(sql).fetchone()[0]

            return {
                "pacientes": um("SELECT COUNT(*) FROM pacientes"),
                "com_alergia": um("SELECT COUNT(DISTINCT paciente_id) FROM alergias"),
                "com_exame_pendente": um(
                    "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE status='pendente'"
                ),
                "com_exame_critico": um(
                    "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE critico=1"
                ),
                "medicacoes_ativas": um("SELECT COUNT(*) FROM medicacoes WHERE ativa=1"),
                "setores": self.setores(),
            }
