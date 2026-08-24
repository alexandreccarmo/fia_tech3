"""
Testes da fundacao do projeto (Etapa 0).

O QUE ESTA SENDO VERIFICADO:
    Que a base sobre a qual todas as outras etapas serao construidas funciona:
    configuracao valida os valores, o catalogo de requisitos esta integro, o
    logging escreve nos tres destinos, a trilha de auditoria produz o trace
    completo e a trava de orcamento realmente bloqueia gasto excedente.

POR QUE TESTAR A FUNDACAO:
    Um erro aqui aparece tarde e mal. Se o guardrail nao registrar nada na
    trilha, so descobriremos na hora de gerar o relatorio - com o prazo em
    cima. Estes testes rodam em menos de um segundo e travam essa classe de
    problema logo na origem.

Rodar com:
    pytest tests/test_fundacao.py -v
"""

from __future__ import annotations

import json

import pytest

# =============================================================================
# CONFIGURACAO
# =============================================================================


class TestSettings:
    def test_caminhos_derivados_ficam_sob_a_raiz(self, cfg_temporario):
        """Todos os diretorios devem ser calculados a partir da raiz do projeto."""
        raiz = cfg_temporario.dir_raiz
        for caminho in (
            cfg_temporario.dir_dados,
            cfg_temporario.dir_logs,
            cfg_temporario.dir_auditoria,
            cfg_temporario.dir_traces,
            cfg_temporario.dir_modelos,
            cfg_temporario.dir_docs,
        ):
            assert caminho.is_relative_to(raiz)

    def test_criar_diretorios_cria_a_arvore_inteira(self, cfg_temporario):
        for caminho in (
            cfg_temporario.dir_dados_brutos,
            cfg_temporario.dir_dados_processados,
            cfg_temporario.dir_dados_sinteticos,
            cfg_temporario.dir_indices,
            cfg_temporario.dir_auditoria,
            cfg_temporario.dir_traces,
            cfg_temporario.dir_adapters,
            cfg_temporario.dir_graficos,
            cfg_temporario.dir_diagramas,
        ):
            assert caminho.is_dir(), f"diretorio nao criado: {caminho}"

    def test_log_level_invalido_e_rejeitado(self):
        from config.settings import Settings

        with pytest.raises(ValueError, match="LOG_LEVEL invalido"):
            Settings(_env_file=None, log_level="VERBOSO")

    def test_log_level_e_normalizado_para_maiusculo(self):
        from config.settings import Settings

        assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"

    def test_overlap_maior_que_o_chunk_e_rejeitado(self):
        """Sobreposicao >= tamanho do chunk causaria laco infinito no splitter."""
        from config.settings import Settings

        with pytest.raises(ValueError, match="deve ser menor"):
            Settings(_env_file=None, rag_chunk_size=500, rag_chunk_overlap=500)

    def test_barra_final_da_url_do_ollama_e_removida(self):
        from config.settings import Settings

        cfg = Settings(_env_file=None, ollama_base_url="http://localhost:11434/")
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_resumo_seguro_nunca_expoe_a_chave_inteira(self):
        """[REQ-3b] A configuracao vai para a auditoria; o segredo, nunca."""
        from config.settings import Settings

        chave = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        resumo = Settings(_env_file=None, openai_api_key=chave).resumo_seguro()

        assert chave not in json.dumps(resumo)
        assert resumo["openai_api_key"].startswith("sk-proj")
        assert resumo["openai_api_key"].endswith("6789")

    def test_resumo_seguro_marca_chave_ausente(self):
        from config.settings import Settings

        resumo = Settings(_env_file=None, openai_api_key="").resumo_seguro()
        assert resumo["openai_api_key"] == "(nao definido)"


# =============================================================================
# CATALOGO DE REQUISITOS
# =============================================================================


class TestRequisitos:
    def test_todos_os_codigos_sao_unicos(self):
        from medgraph.requisitos import CATALOGO

        codigos = [r.codigo for r in CATALOGO]
        assert len(codigos) == len(set(codigos))

    def test_catalogo_cobre_os_quatro_requisitos_obrigatorios(self):
        """Os itens 1 a 4 do enunciado precisam estar todos representados."""
        from medgraph.requisitos import POR_CODIGO

        for codigo in ("REQ-1", "REQ-1a", "REQ-2", "REQ-2a", "REQ-2b",
                       "REQ-3a", "REQ-3b", "REQ-3c", "REQ-4"):
            assert codigo in POR_CODIGO

    def test_catalogo_cobre_os_entregaveis(self):
        from medgraph.requisitos import POR_CODIGO

        for codigo in ("REQ-E1", "REQ-E2", "REQ-E3", "REQ-E4"):
            assert codigo in POR_CODIGO

    def test_tag_inexistente_gera_erro_explicativo(self):
        """Protege contra erro de digitacao em docstring passar despercebido."""
        from medgraph.requisitos import obter

        with pytest.raises(KeyError, match="nao existe no catalogo"):
            obter("REQ-999")

    def test_todo_requisito_aponta_a_origem_no_pdf(self):
        from medgraph.requisitos import CATALOGO

        for requisito in CATALOGO:
            assert requisito.origem.strip(), f"{requisito.codigo} sem origem"
            assert requisito.descricao.strip(), f"{requisito.codigo} sem descricao"


# =============================================================================
# LOGGING  [REQ-3b]
# =============================================================================


class TestLogging:
    def test_tres_destinos_sao_registrados(self, logging_temporario):
        """
        Console, app.log e a trilha JSONL precisam estar todos ativos.

        Contamos por TIPO e nao por quantidade total porque o proprio pytest
        anexa handlers de captura ao logger durante a execucao dos testes.
        """
        import logging
        import logging.handlers

        from medgraph.logging_config import LOGGER_RAIZ, FiltroApenasAuditoria

        handlers = logging.getLogger(LOGGER_RAIZ).handlers

        rotativos = [
            h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        auditoria = [
            h for h in handlers
            if any(isinstance(f, FiltroApenasAuditoria) for f in h.filters)
        ]
        # O pytest injeta LogCaptureHandler (subclasse de StreamHandler) no
        # logger durante a execucao; sao descartados pelo nome da classe.
        console = [
            h for h in handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and "LogCapture" not in type(h).__name__
        ]

        assert len(rotativos) == 1, "faltou o handler de logs/app.log"
        assert len(auditoria) == 1, "faltou o handler da trilha JSONL"
        assert len(console) == 1, "faltou o handler de console"
        assert str(auditoria[0].baseFilename).endswith(".jsonl")

    def test_app_log_recebe_mensagem_comum(self, logging_temporario):
        from medgraph.logging_config import obter_logger

        obter_logger("medgraph.teste").info("mensagem de verificacao")

        conteudo = (logging_temporario.dir_logs / "app.log").read_text(encoding="utf-8")
        assert "mensagem de verificacao" in conteudo

    def test_jsonl_recebe_apenas_eventos_de_auditoria(self, logging_temporario):
        """
        O filtro de auditoria e o que garante que a trilha formal nao vire
        um despejo de debug. Este teste prova que ele funciona nos dois sentidos.
        """
        from medgraph.logging_config import caminho_auditoria_do_dia, obter_logger

        log = obter_logger("medgraph.teste")
        log.info("log comum, NAO deve entrar na trilha")
        log.info("evento formal", extra={"auditoria": True, "trace_id": "abc123"})

        linhas = [
            json.loads(linha)
            for linha in caminho_auditoria_do_dia(logging_temporario)
            .read_text(encoding="utf-8")
            .splitlines()
            if linha.strip()
        ]

        mensagens = [linha["mensagem"] for linha in linhas]
        assert "evento formal" in mensagens
        assert "log comum, NAO deve entrar na trilha" not in mensagens

    def test_jsonl_preserva_campos_extras(self, logging_temporario):
        from medgraph.logging_config import caminho_auditoria_do_dia, obter_logger

        obter_logger("medgraph.teste").info(
            "com dados",
            extra={"auditoria": True, "trace_id": "xyz789", "dados": {"k": 1}},
        )

        registro = json.loads(
            caminho_auditoria_do_dia(logging_temporario)
            .read_text(encoding="utf-8")
            .splitlines()[-1]
        )
        assert registro["trace_id"] == "xyz789"
        assert registro["dados"] == {"k": 1}
        assert "ts" in registro and "nivel" in registro

    def test_obter_logger_normaliza_nomes_soltos(self):
        from medgraph.logging_config import obter_logger

        assert obter_logger("indexacao").name == "medgraph.indexacao"
        assert obter_logger("medgraph.rag.indexar").name == "medgraph.rag.indexar"


# =============================================================================
# TRILHA DE AUDITORIA  [REQ-3b]
# =============================================================================


class TestAuditoria:
    def test_trilha_abre_fecha_e_grava_o_trace(self, logging_temporario):
        from medgraph.auditoria import Desfecho, abrir_trilha

        with abrir_trilha(pergunta="Conduta em sepse?", usuario="dr.silva") as trilha:
            trace_id = trilha.trace_id

        arquivo = logging_temporario.dir_traces / f"{trace_id}.json"
        assert arquivo.exists()

        dossie = json.loads(arquivo.read_text(encoding="utf-8"))
        assert dossie["trace_id"] == trace_id
        assert dossie["usuario"] == "dr.silva"
        assert dossie["desfecho"] == Desfecho.RESPONDIDA.value
        assert dossie["duracao_total_ms"] > 0
        assert dossie["total_eventos"] >= 2  # inicio + fim

    def test_configuracao_vai_para_o_trace_sem_segredo(self, logging_temporario):
        """Reconstruir o contexto de uma resposta exige saber a configuracao."""
        from medgraph.auditoria import abrir_trilha

        with abrir_trilha(pergunta="teste", cfg=logging_temporario) as trilha:
            trace_id = trilha.trace_id

        dossie = json.loads(
            (logging_temporario.dir_traces / f"{trace_id}.json").read_text(encoding="utf-8")
        )
        assert dossie["configuracao"]["llm_provider"] == "eco"
        assert dossie["configuracao"]["openai_api_key"] == "(nao definido)"

    def test_decorator_instrumentar_registra_entrada_saida_e_tempo(self, logging_temporario):
        from medgraph.auditoria import abrir_trilha, instrumentar

        @instrumentar("no_de_teste")
        def no_de_teste(estado: dict) -> dict:
            return {"resposta": "ok"}

        with abrir_trilha(pergunta="teste") as trilha:
            no_de_teste({"pergunta": "x", "paciente_id": "P1"})

            assert "no_de_teste" in trilha.etapas_executadas()
            assert trilha.tempo_por_etapa()["no_de_teste"] >= 0

            inicio = [e for e in trilha.eventos if e.etapa == "no_de_teste"][0]
            assert inicio.dados["chaves_estado"] == ["paciente_id", "pergunta"]

    def test_erro_dentro_de_no_e_auditado_e_re_levantado(self, logging_temporario):
        """Auditoria registra a falha, mas nunca engole a excecao."""
        from medgraph.auditoria import Desfecho, TipoEvento, abrir_trilha, instrumentar

        @instrumentar("no_que_falha")
        def no_que_falha(estado: dict) -> dict:
            raise ValueError("falha proposital")

        with (
            pytest.raises(ValueError, match="falha proposital"),
            abrir_trilha(pergunta="teste", cfg=logging_temporario) as trilha,
        ):
            trilha_ref = trilha
            no_que_falha({})

        erros = trilha_ref.eventos_do_tipo(TipoEvento.ERRO)
        assert erros, "o erro deveria ter sido registrado na trilha"
        assert trilha_ref.desfecho is Desfecho.ERRO

    def test_context_manager_etapa_cronometra_bloco(self, logging_temporario):
        from medgraph.auditoria import abrir_trilha, etapa

        with (
            abrir_trilha(pergunta="teste", cfg=logging_temporario) as trilha,
            etapa("carregar_indice", caminho="/tmp/faiss"),
        ):
            pass

        assert "carregar_indice" in trilha.etapas_executadas()

    def test_no_com_tipo_proprio_continua_na_linha_do_tempo(self, logging_temporario):
        """
        REGRESSAO: a linha do tempo sumia quando um no usava tipo proprio.

        Os nos reais do grafo usam tipos especificos (GUARDRAIL, BANCO, LLM)
        para que a trilha seja pesquisavel por categoria. A versao original de
        `etapas_executadas()` so reconhecia FIM_ETAPA, entao o percurso inteiro
        desaparecia do trace e do painel visual justamente nos nos que mais
        importam auditar.
        """
        from medgraph.auditoria import TipoEvento, abrir_trilha, instrumentar

        @instrumentar("consultar_prontuario", tipo=TipoEvento.BANCO)
        def consultar(estado: dict) -> dict:
            return {"paciente": {"id": "P1"}}

        @instrumentar("guardrail_saida", tipo=TipoEvento.GUARDRAIL)
        def guardrail(estado: dict) -> dict:
            return {"aprovado": True}

        with abrir_trilha(pergunta="teste", cfg=logging_temporario) as trilha:
            consultar({})
            guardrail({})

        assert trilha.etapas_executadas() == ["consultar_prontuario", "guardrail_saida"]
        assert set(trilha.tempo_por_etapa()) == {"consultar_prontuario", "guardrail_saida"}

    def test_mensagens_da_trilha_nao_contem_marcacao_do_rich(self, logging_temporario):
        """
        REGRESSAO: tags de estilo do Rich vazavam para o JSONL e o app.log.

        Alem de sujar a trilha, o problema tinha um efeito pior: com o console
        interpretando colchetes como estilo, uma citacao como [E1] ou um nome
        de etapa entre colchetes era APAGADO da tela. Em um sistema cujo
        requisito e explainability por citacao, isso e inaceitavel. [REQ-3c]
        """
        import json

        from medgraph.auditoria import abrir_trilha, instrumentar
        from medgraph.logging_config import caminho_auditoria_do_dia

        @instrumentar("raciocinio_clinico")
        def raciocinio(estado: dict) -> dict:
            return {"resposta": "Conduta conforme protocolo [P2] e evidencia [E1]."}

        with abrir_trilha(pergunta="teste", cfg=logging_temporario) as trilha:
            raciocinio({})
            trace_id = trilha.trace_id

        for evento in trilha.eventos:
            assert "[dim]" not in evento.mensagem
            assert "[/" not in evento.mensagem

        # A citacao, por outro lado, PRECISA sobreviver intacta ate o disco.
        dossie = (logging_temporario.dir_traces / f"{trace_id}.json").read_text(encoding="utf-8")
        assert "[P2]" in dossie and "[E1]" in dossie

        linhas = caminho_auditoria_do_dia(logging_temporario).read_text(encoding="utf-8")
        assert "[dim]" not in linhas
        assert json.loads(linhas.splitlines()[0])["mensagem"]

    def test_registrar_fora_de_trilha_nao_quebra(self, logging_temporario):
        """Scripts de preparacao de dados rodam sem trilha aberta."""
        from medgraph.auditoria import TipoEvento, registrar, trilha_atual

        assert trilha_atual() is None
        registrar(TipoEvento.BANCO, "consulta fora de trilha")  # nao deve levantar

    def test_texto_longo_e_truncado_com_marcacao(self, logging_temporario):
        """A trilha nunca mente por omissao: o corte e sempre explicito."""
        from medgraph.auditoria import LIMITE_TEXTO_EVENTO, _resumir

        resumido = _resumir("a" * (LIMITE_TEXTO_EVENTO + 500))
        assert "caracteres omitidos" in resumido

    def test_lista_curta_e_gravada_por_inteiro(self, logging_temporario):
        """
        Listas pequenas de valores curtos vao inteiras para a trilha.

        E o caso das chaves do estado e das fontes citadas - justamente o
        que se quer poder auditar depois, item por item. [REQ-3c]
        """
        from medgraph.auditoria import _resumir

        fontes = ["E1:pubmed_31234", "P2:protocolo_sepse", "C1:paciente_0042"]
        assert _resumir(fontes) == fontes

    def test_lista_grande_vira_quantidade_mais_amostra(self, logging_temporario):
        from medgraph.auditoria import MAX_ITENS_LISTA_INTEIRA, _resumir

        resumido = _resumir([f"doc{i}" for i in range(MAX_ITENS_LISTA_INTEIRA + 10)])
        assert resumido["quantidade"] == MAX_ITENS_LISTA_INTEIRA + 10
        assert len(resumido["amostra"]) == 3

    def test_lista_de_textos_longos_e_resumida(self, logging_temporario):
        """Tres abstracts inteiros nao podem ser repetidos em cada evento."""
        from medgraph.auditoria import _resumir

        resumido = _resumir(["x" * 500, "y" * 500])
        assert resumido["quantidade"] == 2
        assert "amostra" in resumido

    def test_redator_de_pii_e_aplicado_antes_de_gravar(self, logging_temporario):
        """[REQ-1a] Nenhum dado pessoal chega ao disco pela trilha."""
        import medgraph.auditoria as modulo

        original = modulo._redator
        try:
            modulo.definir_redator(lambda t: t.replace("Maria Silva", "[NOME]"))
            assert "[NOME]" in modulo._resumir("Paciente Maria Silva internada")
        finally:
            modulo.definir_redator(original)


# =============================================================================
# CONTROLE DE CUSTO  [REQ-3b]
# =============================================================================


class TestCusto:
    def test_calculo_confere_com_a_tabela_de_precos(self):
        from medgraph.llm.custo import ContadorCusto

        # gpt-4o-mini: US$ 0,15 por 1M de entrada e US$ 0,60 por 1M de saida.
        custo = ContadorCusto.calcular_custo("gpt-4o-mini", 1_000_000, 1_000_000)
        assert custo == pytest.approx(0.75)

    def test_modelo_local_nao_gera_custo(self):
        """
        O custo zero é decidido pelo PROVEDOR, não pelo nome do modelo.

        Uma lista de nomes deixaria de fora o primeiro nome novo que alguém
        escolhesse no .env — e o contador atribuiria a ele o preço da OpenAI,
        inflando a tabela do relatório com um gasto que nunca existiu.
        """
        from medgraph.llm.custo import ContadorCusto

        for nome in ("medgraph", "medgraph-base", "medgraph-v2-experimental"):
            assert ContadorCusto.calcular_custo(
                nome, 500_000, 500_000, provedor="ollama"
            ) == 0.0

    def test_provedor_pago_gera_custo(self):
        from medgraph.llm.custo import ContadorCusto

        assert ContadorCusto.calcular_custo(
            "gpt-4o-mini", 1_000_000, 0, provedor="openai"
        ) == pytest.approx(0.15)

    def test_modelo_desconhecido_usa_preco_padrao(self):
        """Melhor superestimar do que assumir zero e furar o orcamento."""
        from medgraph.llm.custo import ContadorCusto

        assert ContadorCusto.calcular_custo("modelo-que-nao-existe", 1_000_000, 0) > 0

    def test_orcamento_bloqueia_chamada_acima_do_teto(self, logging_temporario):
        from medgraph.llm.custo import OrcamentoExcedidoError, reiniciar_contador

        c = reiniciar_contador(limite_usd=0.10)
        c.registrar_uso("gpt-4o-mini", tokens_entrada=600_000, tokens_saida=0)  # US$ 0,09

        c.verificar_orcamento(custo_previsto_usd=0.005)  # ainda cabe
        with pytest.raises(OrcamentoExcedidoError, match="Orcamento da sessao esgotado"):
            c.verificar_orcamento(custo_previsto_usd=0.50)

    def test_mensagem_de_bloqueio_ensina_como_resolver(self, logging_temporario):
        from medgraph.llm.custo import OrcamentoExcedidoError, reiniciar_contador

        c = reiniciar_contador(limite_usd=0.0)
        with pytest.raises(OrcamentoExcedidoError) as info:
            c.verificar_orcamento(custo_previsto_usd=0.01)

        texto = str(info.value)
        assert "MAX_CUSTO_USD_SESSAO" in texto
        assert "ollama" in texto

    def test_agregacao_por_modelo(self, logging_temporario):
        from medgraph.llm.custo import reiniciar_contador

        c = reiniciar_contador(limite_usd=10.0)
        c.registrar_uso("gpt-4o-mini", 1000, 200, origem="avaliacao")
        c.registrar_uso("gpt-4o-mini", 2000, 400, origem="avaliacao")
        c.registrar_uso("medgraph", 5000, 900, origem="grafo", provedor="ollama")

        por_modelo = c.por_modelo()
        assert por_modelo["gpt-4o-mini"]["chamadas"] == 2
        assert por_modelo["gpt-4o-mini"]["tokens_entrada"] == 3000
        assert por_modelo["medgraph"]["custo_usd"] == 0.0
        assert c.total_chamadas == 3

    def test_uso_gera_evento_na_trilha(self, logging_temporario):
        """[REQ-3b] Consumo tambem e informacao auditavel."""
        from medgraph.auditoria import TipoEvento, abrir_trilha
        from medgraph.llm.custo import reiniciar_contador

        c = reiniciar_contador(limite_usd=10.0)
        with abrir_trilha(pergunta="teste") as trilha:
            c.registrar_uso("gpt-4o-mini", 1000, 100, origem="teste")

        assert trilha.eventos_do_tipo(TipoEvento.CUSTO)

    def test_tabela_resumo_tem_total_e_saldo(self, logging_temporario):
        from medgraph.llm.custo import reiniciar_contador

        c = reiniciar_contador(limite_usd=1.0)
        c.registrar_uso("gpt-4o-mini", 1000, 100)

        tabela = c.tabela_resumo()
        assert "TOTAL" in tabela and "Saldo" in tabela

    def test_estimativa_de_tokens_e_proporcional_ao_texto(self):
        from medgraph.llm.custo import estimar_tokens

        curto = estimar_tokens("Qual a conduta em sepse?")
        longo = estimar_tokens("Qual a conduta em sepse? " * 100)
        assert 0 < curto < longo


# =============================================================================
# POLITICAS  [REQ-3a]
# =============================================================================


class TestPoliticas:
    def test_arquivo_de_politicas_existe_e_e_yaml_valido(self):
        import yaml

        from config.settings import obter_settings

        caminho = obter_settings().caminho_politicas
        assert caminho.exists(), "config/politicas.yaml nao encontrado"

        politicas = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        for secao in ("identidade", "escopo", "entrada", "saida", "risco", "textos"):
            assert secao in politicas, f"secao ausente em politicas.yaml: {secao}"

    def test_regexes_das_politicas_compilam(self):
        """Regex quebrada so apareceria em producao, no primeiro bloqueio."""
        import re

        import yaml

        from config.settings import obter_settings

        politicas = yaml.safe_load(
            obter_settings().caminho_politicas.read_text(encoding="utf-8")
        )

        for padrao in politicas["entrada"]["padroes_bloqueio"]:
            re.compile(padrao["regex"], re.IGNORECASE)
        for padrao in politicas["saida"]["padroes_posologia"]:
            re.compile(padrao, re.IGNORECASE)
        re.compile(politicas["saida"]["formato_citacao"])

    def test_conduta_terapeutica_sempre_exige_validacao_humana(self):
        """[REQ-3a] O item mais sensivel do enunciado, verificado por teste."""
        import yaml

        from config.settings import obter_settings

        politicas = yaml.safe_load(
            obter_settings().caminho_politicas.read_text(encoding="utf-8")
        )
        conduta = next(
            i for i in politicas["escopo"]["intencoes_permitidas"]
            if i["id"] == "conduta_terapeutica"
        )
        assert conduta.get("sempre_validacao_humana") is True

    def test_assistente_nao_atende_pacientes(self):
        import yaml

        from config.settings import obter_settings

        politicas = yaml.safe_load(
            obter_settings().caminho_politicas.read_text(encoding="utf-8")
        )
        assert politicas["identidade"]["atende_pacientes"] is False
