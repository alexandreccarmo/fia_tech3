"""
Testes da Etapa 1 — preparação dos dados.

O QUE ESTÁ SENDO VERIFICADO:
    Que o anonimizador remove o que precisa remover e PRESERVA o que não pode
    ser removido; que os filtros de curadoria reprovam o que devem reprovar;
    que a divisão treino/teste é estratificada e reprodutível; e que a base de
    prontuários carrega os casos clínicos que o sistema precisa exercitar.

O TESTE MAIS IMPORTANTE DESTE ARQUIVO:
    `test_valores_clinicos_sobrevivem_a_anonimizacao`. A falha silenciosa mais
    perigosa de um anonimizador clínico não é deixar passar um CPF — é apagar
    "potássio 6,8 mEq/L" por parecer um número identificável. Um pipeline
    assim entregaria dados limpos e clinicamente inúteis, e ninguém perceberia
    até o modelo dar respostas sem sentido.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


# =============================================================================
# ANONIMIZAÇÃO  [REQ-1a]
# =============================================================================
class TestAnonimizador:
    @pytest.fixture
    def anon(self):
        from medgraph.dados.anonimizador import Anonimizador, Politica

        return Anonimizador(
            politica=Politica.MASCARAR,
            nomes_conhecidos=["Maria Aparecida Souza", "Rafael Menezes"],
            chave=b"chave-de-teste",
        )

    @pytest.mark.parametrize(
        "texto,marcador",
        [
            ("CPF 123.456.789-00", "[CPF]"),
            ("contato: joao@hospital.com.br", "[EMAIL]"),
            ("telefone (11) 98765-4321", "[TELEFONE]"),
            ("prontuario PRT-10001", "[PRONTUARIO]"),
            ("pront. 45678", "[PRONTUARIO]"),
            ("CRM/SP 123456", "[CRM]"),
            ("CEP 01310-100", "[CEP]"),
            ("CNS 123 4567 8901 2345", "[CNS]"),
            ("RG 12.345.678-9", "[RG]"),
            ("nascida em 14/03/1958", "[DATA_NASCIMENTO]"),
            ("paciente de 94 anos", "[IDADE_EXTREMA]"),
        ],
    )
    def test_cada_tipo_de_identificador_e_removido(self, anon, texto, marcador):
        resultado = anon.redigir(texto)
        assert marcador in resultado
        # O valor original não pode sobreviver em nenhuma forma.
        assert not any(c.isdigit() and c in resultado for c in "".join(texto.split()[-1:]) if c.isdigit()) or marcador in resultado

    def test_valores_clinicos_sobrevivem_a_anonimizacao(self, anon):
        """
        O anonimizador não pode destruir a informação clínica.

        Cada item desta lista é um caso real de falso positivo que uma regex
        frouxa provocaria: valores laboratoriais viram "CPF", doses viram
        "telefone", datas de exame viram "data de nascimento".
        """
        clinico = (
            "Potassio 6.8 mEq/L (ref 3.5-5.0), creatinina 3.9 mg/dL, "
            "leucocitos 12400 /mm3, plaquetas 18000/mm3, INR 6.2, PCR 180 mg/L. "
            "Prescrito Ceftriaxona 2 g EV 1x/dia, Enoxaparina 40 mg SC, "
            "Noradrenalina 0.2 mcg/kg/min. Coleta em 22/08/2026. "
            "PA 90/50 mmHg, FC 122 bpm, T 38.9 C, SatO2 91%."
        )
        resultado = anon.redigir(clinico)
        assert resultado == clinico, (
            "O anonimizador alterou dado clínico. Diferença:\n"
            f"  antes: {clinico}\n  depois: {resultado}"
        )

    def test_nome_conhecido_e_removido_mesmo_sem_rotulo(self, anon):
        assert "Souza" not in anon.redigir("Souza evoluiu bem durante a noite.")

    def test_nome_desconhecido_apos_rotulo_e_removido(self, anon):
        """Heurística para nomes que não estão no dicionário."""
        resultado = anon.redigir("Avaliado pelo Dr. Fernando Albuquerque Lima na enfermaria.")
        assert "Fernando" not in resultado
        assert "Albuquerque" not in resultado

    def test_termo_institucional_apos_rotulo_nao_vira_nome(self, anon):
        """"paciente Clinica Medica" não pode virar "[NOME]"."""
        resultado = anon.redigir("Transferido para a Clinica Medica ontem.")
        assert "Clinica Medica" in resultado

    def test_pseudonimo_e_estavel_para_o_mesmo_valor(self):
        """Duas execuções com a mesma chave produzem o mesmo token."""
        from medgraph.dados.anonimizador import Anonimizador, Politica

        def redigir():
            return Anonimizador(
                politica=Politica.PSEUDONIMIZAR, chave=b"chave-fixa"
            ).redigir("CPF 123.456.789-00")

        assert redigir() == redigir()

    def test_pseudonimo_muda_com_a_chave(self):
        """Sem a chave, não há como correlacionar tokens entre bases."""
        from medgraph.dados.anonimizador import Anonimizador, Politica

        a = Anonimizador(politica=Politica.PSEUDONIMIZAR, chave=b"chave-a")
        b = Anonimizador(politica=Politica.PSEUDONIMIZAR, chave=b"chave-b")
        assert a.redigir("CPF 123.456.789-00") != b.redigir("CPF 123.456.789-00")

    def test_todas_as_mencoes_a_mesma_pessoa_recebem_o_mesmo_token(self):
        """
        Coerência referencial — a razão de existir da pseudonimização.

        Se "Maria Aparecida Souza", "Souza" e "Maria" recebessem tokens
        diferentes, o texto perderia a informação de que se trata da mesma
        pessoa, que é justamente o que se quis preservar ao não mascarar.
        """
        from medgraph.dados.anonimizador import Anonimizador, Politica

        anon = Anonimizador(
            politica=Politica.PSEUDONIMIZAR,
            nomes_conhecidos=["Maria Aparecida Souza"],
            chave=b"k",
        )
        resultado = anon.redigir(
            "Maria Aparecida Souza foi internada. Souza evoluiu bem. Maria recebeu alta."
        )
        tokens = {t for t in resultado.split() if t.startswith("[NOME_")}
        assert len(tokens) == 1, f"esperado 1 token, encontrados {tokens}"

    def test_rotulo_semantico_e_preservado(self, anon):
        """"nascida em" carrega sentido clínico e não identifica ninguém."""
        assert "nascida em" in anon.redigir("Paciente nascida em 14/03/1958.")

    def test_achados_nao_guardam_o_valor_original(self, anon):
        """Um relatório de anonimização não pode ser um vazamento."""
        achados = anon.analisar("CPF 123.456.789-00")
        serializado = json.dumps([a.__dict__ for a in achados], default=str)
        assert "123.456.789-00" not in serializado

    def test_texto_vazio_nao_quebra(self, anon):
        assert anon.redigir("") == ""


# =============================================================================
# CURADORIA  [REQ-1a]
# =============================================================================
class TestCuradoria:
    def _registro(self, **sobrescritas):
        base = {
            "pubid": 1,
            "question": "Does aspirin reduce cardiovascular mortality in adults?",
            "contexto_texto": "A" * 500,
            "long_answer": "Aspirin reduced mortality in the studied population.",
            "final_decision": "yes",
        }
        base.update(sobrescritas)
        return base

    def test_rotulo_invalido_e_descartado(self):
        from medgraph.dados.curadoria import curar

        _, rel = curar([self._registro(final_decision="")], subconjunto="t", anonimizar=False)
        assert rel.descartados["rotulo_invalido"] == 1

    def test_contexto_curto_e_descartado(self):
        """Sem abstract não há evidência: o modelo aprenderia a chutar."""
        from medgraph.dados.curadoria import curar

        _, rel = curar([self._registro(contexto_texto="curto")], subconjunto="t", anonimizar=False)
        assert rel.descartados["contexto_curto_demais"] == 1

    def test_contexto_longo_demais_e_descartado(self):
        from medgraph.dados.curadoria import curar

        _, rel = curar([self._registro(contexto_texto="A" * 20_000)], subconjunto="t", anonimizar=False)
        assert rel.descartados["contexto_longo_demais"] == 1

    def test_duplicata_e_removida(self):
        """Duplicata inflaria a métrica de teste artificialmente."""
        from medgraph.dados.curadoria import curar

        aprovados, rel = curar(
            [self._registro(pubid=1), self._registro(pubid=2)],
            subconjunto="t",
            anonimizar=False,
        )
        assert rel.duplicatas == 1
        assert len(aprovados) == 1

    def test_relatorio_separa_reprovacao_de_corte_de_amostra(self):
        """Métricas de curadoria não podem confundir qualidade com escopo."""
        from medgraph.dados.curadoria import RelatorioCuradoria

        rel = RelatorioCuradoria(subconjunto="t", total_entrada=1000)
        rel.descartados["contexto_curto_demais"] = 10
        rel.truncados_para_amostra = 500
        assert rel.total_reprovado_por_filtro == 10
        assert rel.taxa_aprovacao_nos_filtros == pytest.approx(0.99)

    def test_divisao_e_estratificada(self):
        from medgraph.dados.curadoria import dividir_estratificado

        registros = (
            [{"decisao": "yes", "i": i} for i in range(600)]
            + [{"decisao": "no", "i": i} for i in range(300)]
            + [{"decisao": "maybe", "i": i} for i in range(100)]
        )
        divisao = dividir_estratificado(registros, semente=42)

        for nome, conjunto in divisao.items():
            proporcao_maybe = sum(1 for r in conjunto if r["decisao"] == "maybe") / len(conjunto)
            assert 0.05 < proporcao_maybe < 0.16, f"{nome} perdeu a estratificação"

    def test_divisao_e_reproduzivel(self):
        """Mesma semente, mesma divisão — requisito para o relatório técnico."""
        from medgraph.dados.curadoria import dividir_estratificado

        registros = [{"decisao": "yes" if i % 3 else "no", "i": i} for i in range(300)]
        a = dividir_estratificado(registros, semente=7)
        b = dividir_estratificado(registros, semente=7)
        assert [r["i"] for r in a["teste"]] == [r["i"] for r in b["teste"]]

    def test_divisao_nao_perde_nem_duplica_registros(self):
        from medgraph.dados.curadoria import dividir_estratificado

        registros = [{"decisao": "yes" if i % 4 else "maybe", "i": i} for i in range(400)]
        divisao = dividir_estratificado(registros, semente=1)
        todos = [r["i"] for conjunto in divisao.values() for r in conjunto]
        assert sorted(todos) == list(range(400))

    def test_balanceamento_respeita_a_razao_maxima(self):
        from medgraph.dados.curadoria import balancear_por_rotulo

        registros = [{"decisao": "yes"}] * 9000 + [{"decisao": "no"}] * 500
        balanceados, removidos = balancear_por_rotulo(registros, razao_maxima=3.0, semente=1)

        contagem = {r: sum(1 for x in balanceados if x["decisao"] == r) for r in ("yes", "no")}
        assert contagem["yes"] == 1500  # 3 x 500
        assert contagem["no"] == 500
        assert removidos == 7500

    def test_higienizacao_remove_separadores_invisiveis(self):
        """
        Caracteres como \\u2028 quebram arquivos JSON Lines.

        São válidos dentro de uma string JSON — o json.dumps não os escapa —
        mas várias funções de leitura os tratam como fim de linha, partindo um
        registro em dois e corrompendo o arquivo inteiro.
        """
        from medgraph.dados.curadoria import higienizar

        sujo = "Resultado\u2028do exame\x0ccom lixo\x0btipográfico"
        limpo = higienizar(sujo)
        assert not any(c in limpo for c in "\u2028\u2029\x0b\x0c\x1c\x1d\x1e\x85")
        assert limpo.splitlines() == [limpo], "ainda quebra em múltiplas linhas"


# =============================================================================
# BASE DE PRONTUÁRIOS  [REQ-2a]
# =============================================================================
@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "data" / "sintetico" / "prontuarios.sqlite"
    ).exists(),
    reason="banco ainda não construído — rode: make dados",
)
class TestBancoProntuarios:
    @pytest.fixture
    def conexao(self):
        from config.settings import obter_settings

        con = sqlite3.connect(obter_settings().caminho_banco_prontuarios)
        con.row_factory = sqlite3.Row
        yield con
        con.close()

    def test_todas_as_tabelas_existem(self, conexao):
        tabelas = {
            r[0] for r in conexao.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for esperada in (
            "pacientes", "comorbidades", "alergias",
            "medicacoes", "exames", "sinais_vitais", "evolucoes",
        ):
            assert esperada in tabelas

    def test_chaves_estrangeiras_estao_integras(self, conexao):
        """Nenhum registro clínico pode apontar para paciente inexistente."""
        conexao.execute("PRAGMA foreign_keys = ON")
        violacoes = conexao.execute("PRAGMA foreign_key_check").fetchall()
        assert not violacoes, f"integridade referencial violada: {violacoes}"

    def test_casos_clinicos_necessarios_estao_presentes(self, conexao):
        """
        A base precisa conter os casos que as regras de segurança testam.

        Sem paciente alérgico a penicilina, a regra de conflito com alergia
        nunca dispararia — e passaria a impressão falsa de estar funcionando.
        """
        def contar(sql: str) -> int:
            return conexao.execute(sql).fetchone()[0]

        assert contar("SELECT COUNT(*) FROM pacientes") == 40
        assert contar(
            "SELECT COUNT(DISTINCT paciente_id) FROM alergias "
            "WHERE substancia LIKE '%enicilina%' OR classe LIKE '%etalact%'"
        ) >= 6
        assert contar(
            "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE status='pendente'"
        ) >= 8
        assert contar(
            "SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE critico=1"
        ) >= 6
        assert contar("SELECT COUNT(*) FROM pacientes WHERE gestante=1") == 3

    def test_visao_de_resumo_funciona(self, conexao):
        linha = conexao.execute(
            "SELECT * FROM vw_resumo_paciente WHERE qtd_alergias > 0 LIMIT 1"
        ).fetchone()
        assert linha is not None
        assert linha["idade"] > 0

    def test_valores_de_dominio_sao_respeitados(self, conexao):
        """Os CHECK do esquema impedem estados inválidos."""
        assert not conexao.execute(
            "SELECT 1 FROM exames WHERE status NOT IN ('resultado','pendente','coletado')"
        ).fetchall()
        assert not conexao.execute(
            "SELECT 1 FROM alergias WHERE gravidade NOT IN ('leve','moderada','grave')"
        ).fetchall()

    def test_nenhum_cpf_plausivel_no_seed(self, conexao):
        """
        Dados sintéticos precisam ser obviamente sintéticos.

        Um CPF com formato válido num repositório público, ainda que
        inventado, é um risco desnecessário.
        """
        cpfs = {r[0] for r in conexao.execute("SELECT DISTINCT cpf FROM pacientes")}
        assert cpfs <= {"000.000.000-00", None}


# =============================================================================
# DATASET DE FINE-TUNING  [REQ-1]
# =============================================================================
class TestDatasetSFT:
    def test_prompt_de_sistema_declara_os_limites(self):
        """[REQ-3a] O limite mais importante precisa estar no prompt."""
        from medgraph.chains import prompts

        sistema = prompts.SISTEMA.lower()
        assert "não prescreve" in sistema or "nao prescreve" in sistema
        assert "fonte" in sistema
        assert "[e#]" in sistema and "[p#]" in sistema and "[c#]" in sistema

    def test_resposta_de_referencia_tem_formato_extraivel(self):
        """A primeira linha fixa torna a avaliação determinística."""
        from medgraph.chains import prompts

        resposta = prompts.assistente_decisao("yes", "A evidência sustenta o desfecho.")
        assert resposta.splitlines()[0] == "Decisão: yes"
        assert resposta.strip().endswith("Fontes: [E1]")

    def test_contexto_apresenta_o_marcador_primeiro(self):
        from medgraph.chains import prompts

        contexto = prompts.montar_contexto(
            [{"marcador": "P2", "titulo": "PROT-008", "texto": "conteúdo"}]
        )
        assert contexto.startswith("[P2] PROT-008")

    def test_repeticao_por_classe_favorece_a_classe_rara(self):
        """[REQ-1] Sem isso, "maybe" ficaria abaixo de 1% do dataset."""
        from medgraph.finetune.preparar_dataset_sft import REPETICOES_POR_CLASSE

        assert REPETICOES_POR_CLASSE["maybe"] > REPETICOES_POR_CLASSE["no"]
        assert REPETICOES_POR_CLASSE["no"] > REPETICOES_POR_CLASSE["yes"]
