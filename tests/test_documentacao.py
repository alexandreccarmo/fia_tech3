"""
Testes das afirmações da documentação.

O QUE ESTÁ SENDO VERIFICADO:
    Que os números escritos nas docstrings, no README e no relatório técnico
    correspondem ao que o código realmente faz.

POR QUE ISSO É UM TESTE, E NÃO UMA REVISÃO:
    Este arquivo nasceu de um erro real. O grafo foi planejado com doze nós;
    durante a implementação surgiram mais dois — `responder_recusa` e
    `degradar_resposta` —, e a prosa nunca foi atualizada. Cinco arquivos
    diferentes passaram a afirmar "doze nós" sobre um grafo de quatorze.

    Nenhum teste falhou, nenhum linter reclamou, e o erro sobreviveu a várias
    revisões porque documentação não é executada. Uma afirmação numérica que
    ninguém verifica envelhece errada — é só questão de tempo.

    Tornar as afirmações falsificáveis é a única forma de mantê-las honestas.
    Um projeto acadêmico que será lido por um avaliador não pode ter números
    decorativos.

O QUE ESTE ARQUIVO NÃO FAZ:
    Não verifica prosa, argumentação ou julgamento — só o que é contável.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from config.settings import RAIZ_PROJETO, obter_settings


# =============================================================================
# ESTRUTURA DO GRAFO
# =============================================================================
class TestAfirmacoesSobreOGrafo:
    @pytest.fixture(scope="class")
    def grafo(self):
        from medgraph.grafo.construir import montar_grafo

        return montar_grafo()

    def test_numero_de_nos_bate_com_a_documentacao(self, grafo):
        """
        REGRESSÃO — cinco arquivos afirmavam "doze nós" sobre um grafo de catorze.

        O grafo foi planejado com doze; `responder_recusa` e `degradar_resposta`
        surgiram durante a implementação, e a prosa não acompanhou.
        """
        total = len(grafo.nodes)

        arquivos = [
            RAIZ_PROJETO / "README.md",
            RAIZ_PROJETO / "src/medgraph/grafo/estado.py",
            RAIZ_PROJETO / "src/medgraph/grafo/construir.py",
            RAIZ_PROJETO / "src/medgraph/grafo/nos.py",
            RAIZ_PROJETO / "scripts/07_rodar_grafo.py",
        ]

        # Números por extenso que apareceriam se a contagem estivesse errada.
        por_extenso = {
            10: "dez", 11: "onze", 12: "doze", 13: "treze",
            14: "quatorze", 15: "quinze", 16: "dezesseis",
        }
        errados = {n: e for n, e in por_extenso.items() if n != total}

        problemas: list[str] = []
        for arquivo in arquivos:
            if not arquivo.exists():
                continue
            texto = arquivo.read_text(encoding="utf-8").lower()
            for numero, extenso in errados.items():
                for forma in (f"{numero} nós", f"{numero} nos ", f"{extenso} nós", f"{extenso} nos "):
                    if forma in texto:
                        problemas.append(f"{arquivo.name}: diz '{forma.strip()}' (real: {total})")

        assert not problemas, "contagem de nós divergente:\n  " + "\n  ".join(problemas)

    def test_quantidade_de_bifurcacoes_condicionais(self, grafo):
        """
        O fluxo tem quatro pontos de decisão. É o que a documentação afirma e o
        que o diagrama mostra.
        """
        compilado = grafo.compile()
        origens_condicionais = {
            aresta.source for aresta in compilado.get_graph().edges if aresta.conditional
        }
        assert origens_condicionais == {
            "guardrail_entrada",
            "classificar_intencao",
            "guardrail_saida",
            "emitir_alertas",
        }

    def test_existe_exatamente_um_ciclo(self, grafo):
        """
        A reescrita é a única aresta que volta. Um segundo ciclo não declarado
        seria um caminho de execução que ninguém desenhou.
        """
        compilado = grafo.compile()
        ordem = [
            "guardrail_entrada", "classificar_intencao", "consultar_prontuario",
            "recuperar_evidencia", "raciocinio_clinico", "regras_clinicas",
            "guardrail_saida", "reescrever", "degradar_resposta", "triagem_risco",
            "emitir_alertas", "aguardar_validacao", "montar_resposta",
        ]
        posicao = {nome: i for i, nome in enumerate(ordem)}

        retrocessos = [
            (a.source, a.target)
            for a in compilado.get_graph().edges
            if a.source in posicao and a.target in posicao
            and posicao[a.target] < posicao[a.source]
        ]
        assert retrocessos == [("reescrever", "raciocinio_clinico")]


# =============================================================================
# CONTAGENS DO CORPUS E DA CONFIGURAÇÃO
# =============================================================================
class TestAfirmacoesSobreOsDados:
    def test_categorias_de_pii(self):
        """O relatório afirma onze categorias de identificador."""
        from medgraph.dados.anonimizador import TipoPII

        assert len(list(TipoPII)) == 11

    def test_intencoes_e_padroes_de_bloqueio(self):
        from medgraph.guardrails import politicas

        pol = politicas.carregar()
        assert len(pol.intencoes_permitidas()) == 5, "o fluxo documenta cinco intenções"
        assert len(pol.padroes_bloqueio) == 4, "o relatório documenta quatro padrões de bloqueio"

    def test_corpus_sintetico_tem_o_volume_documentado(self):
        cfg = obter_settings()
        protocolos = list((cfg.dir_dados_sinteticos / "protocolos").glob("PROT-*.md"))
        documentos = list((cfg.dir_dados_sinteticos / "modelos_documentos").glob("DOC-*.md"))

        assert len(protocolos) == 15
        assert len(documentos) == 10

        with (cfg.dir_dados_sinteticos / "faq_medicos.jsonl").open(encoding="utf-8") as arquivo:
            assert sum(1 for linha in arquivo if linha.strip()) == 200

    def test_catalogo_tem_treze_requisitos(self):
        from medgraph.requisitos import CATALOGO

        assert len(CATALOGO) == 13


@pytest.mark.skipif(
    not (RAIZ_PROJETO / "data" / "sintetico" / "prontuarios.sqlite").exists(),
    reason="banco ainda não construído — rode: make dados",
)
class TestAfirmacoesSobreOBanco:
    def test_sete_tabelas_e_quarenta_pacientes(self):
        cfg = obter_settings()
        conexao = sqlite3.connect(f"file:{cfg.caminho_banco_prontuarios}?mode=ro", uri=True)
        try:
            tabelas = [
                r[0] for r in conexao.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            assert len(tabelas) == 7
            assert conexao.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0] == 40
        finally:
            conexao.close()


# =============================================================================
# INTEGRIDADE DOS DOCUMENTOS
# =============================================================================
class TestIntegridadeDosDocumentos:
    def test_relatorio_nao_tem_marcador_por_substituir(self):
        """
        Um `{{ALGO}}` sobrando apareceria no meio do relatório entregue.

        O gerador avisa, mas o aviso vai para o terminal de quem gerou — e
        ninguém relê um documento de setecentas linhas antes de entregar.
        """
        caminho = RAIZ_PROJETO / "docs" / "relatorio_tecnico.md"
        if not caminho.exists():
            pytest.skip("relatório ainda não gerado — rode: make relatorio")

        sobraram = re.findall(r"\{\{[A-Z_]+\}\}", caminho.read_text(encoding="utf-8"))
        assert not sobraram, f"marcadores não substituídos: {set(sobraram)}"

    def test_documentos_referenciados_pelo_readme_existem(self):
        """Link quebrado num README é o primeiro sinal de documentação abandonada."""
        texto = (RAIZ_PROJETO / "README.md").read_text(encoding="utf-8")

        # Apenas links relativos para arquivos do próprio repositório.
        alvos = {
            alvo.split("#")[0]
            for alvo in re.findall(r"\]\(([^)#][^)]*)\)", texto)
            if not alvo.startswith(("http", "mailto:"))
        }
        ausentes = [a for a in sorted(alvos) if a and not (RAIZ_PROJETO / a).exists()]
        assert not ausentes, f"o README aponta para arquivos inexistentes: {ausentes}"

    def test_notebooks_do_colab_sao_json_valido(self):
        """Um .ipynb corrompido só falha quando alguém tenta abri-lo no Colab."""
        import json

        notebooks = list((RAIZ_PROJETO / "notebooks" / "colab").glob("*.ipynb"))
        assert len(notebooks) == 2

        for caminho in notebooks:
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
            assert conteudo["nbformat"] == 4
            assert conteudo["cells"], f"{caminho.name} não tem células"
