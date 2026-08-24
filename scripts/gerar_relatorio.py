#!/usr/bin/env python
"""
[REQ-E3] Gerador do relatório técnico.

O QUE FAZ:
    Monta `docs/relatorio_tecnico.md` combinando a narrativa — escrita à mão,
    em `docs/relatorio_base.md` — com os NÚMEROS lidos diretamente dos
    artefatos que o pipeline produziu.

POR QUE GERAR EM VEZ DE ESCREVER À MÃO:
    Um relatório técnico com números digitados manualmente começa correto e
    envelhece errado. Basta reexecutar a avaliação com outro tamanho de
    amostra, ou refazer a curadoria, para que a tabela do documento deixe de
    corresponder aos arquivos do repositório — e ninguém percebe, porque as
    duas coisas vivem em lugares diferentes.

    Aqui o texto explica e os arquivos informam. Se um número mudar, basta
    rodar `make relatorio`; se um artefato não existir, o documento diz
    explicitamente qual comando produzi-lo, em vez de exibir um número velho.

    A narrativa continua sendo escrita por pessoas: só os valores são lidos.

Uso:
    make relatorio
    python scripts/gerar_relatorio.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from config.settings import obter_settings  # noqa: E402

AUSENTE = "_(artefato não gerado — rode `{comando}`)_"


def _ler_json(caminho: Path) -> dict[str, Any] | None:
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _contar_linhas(caminho: Path) -> int:
    if not caminho.exists():
        return 0
    with caminho.open(encoding="utf-8") as arquivo:
        return sum(1 for linha in arquivo if linha.strip())


# =============================================================================
# BLOCOS DE DADOS
# =============================================================================
def bloco_dados(cfg) -> str:
    curadoria = _ler_json(cfg.dir_dados_processados / "relatorio_curadoria.json")
    sft = _ler_json(cfg.dir_dados_processados / "relatorio_sft.json")

    if not curadoria or not sft:
        return AUSENTE.format(comando="make dados")

    linhas = ["#### Curadoria do PubMedQA", "", "| Subconjunto | Entrada | Reprovados nos filtros | Taxa de aprovação | Saída |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in curadoria["relatorios"]:
        linhas.append(
            f"| `{r['subconjunto']}` | {r['total_entrada']:,} | {r['reprovados_por_filtro']:,} "
            f"| {r['taxa_aprovacao_nos_filtros']:.1%} | {r['total_saida']:,} |"
        )

    divisao = curadoria["divisao_anotado"]
    linhas += [
        "",
        "#### Divisão do conjunto anotado por especialistas",
        "",
        "| Conjunto | Exemplos | Uso |",
        "| --- | ---: | --- |",
        f"| Treino | {divisao['treino']} | fine-tuning |",
        f"| Validação | {divisao['validacao']} | acompanhamento da perda durante o treino |",
        f"| **Teste** | **{divisao['teste']}** | **avaliação — nunca visto no treino** |",
        "",
        "#### Dataset de fine-tuning",
        "",
        "| Origem | Exemplos |",
        "| --- | ---: |",
    ]
    for origem, quantidade in sorted(sft["por_origem"].items(), key=lambda x: -x[1]):
        linhas.append(f"| {origem} | {quantidade:,} |")
    linhas.append(f"| **Total** | **{sft['total']:,}** |")

    linhas += [
        "",
        "| Rótulo | Exemplos | Proporção |",
        "| --- | ---: | ---: |",
    ]
    for rotulo, quantidade in sorted(sft["por_rotulo"].items(), key=lambda x: -x[1]):
        proporcao = sft["proporcao_por_rotulo"][rotulo]
        linhas.append(f"| `{rotulo}` | {quantidade:,} | {proporcao:.1%} |")

    linhas += [
        "",
        f"Repetição por classe aplicada ao conjunto de especialista: "
        f"`{sft['repeticoes_aplicadas']}`. Sem ela, `maybe` representaria menos de 1% do "
        f"dataset e o modelo nunca a preveria.",
        "",
        f"Tamanho médio por exemplo: {sft['caracteres_por_exemplo']['media']:,} caracteres "
        f"(~{sft['tokens_estimados_total'] // sft['total']} tokens). "
        f"Total estimado: ~{sft['tokens_estimados_total'] / 1e6:.1f} milhões de tokens por época.",
    ]
    return "\n".join(linhas)


def bloco_prontuarios(cfg) -> str:
    if not cfg.caminho_banco_prontuarios.exists():
        return AUSENTE.format(comando="make dados")

    conexao = sqlite3.connect(f"file:{cfg.caminho_banco_prontuarios}?mode=ro", uri=True)
    try:
        def um(sql: str) -> int:
            return conexao.execute(sql).fetchone()[0]

        indicadores = [
            ("Pacientes", um("SELECT COUNT(*) FROM pacientes")),
            ("Comorbidades registradas", um("SELECT COUNT(*) FROM comorbidades")),
            ("Alergias registradas", um("SELECT COUNT(*) FROM alergias")),
            ("Medicações ativas", um("SELECT COUNT(*) FROM medicacoes WHERE ativa=1")),
            ("Exames", um("SELECT COUNT(*) FROM exames")),
            ("Aferições de sinais vitais", um("SELECT COUNT(*) FROM sinais_vitais")),
            ("Evoluções clínicas", um("SELECT COUNT(*) FROM evolucoes")),
        ]
        casos = [
            ("Pacientes com alguma alergia", um("SELECT COUNT(DISTINCT paciente_id) FROM alergias")),
            (
                "Pacientes com alergia a betalactâmico",
                um("SELECT COUNT(DISTINCT paciente_id) FROM alergias "
                   "WHERE substancia LIKE '%enicilina%' OR classe LIKE '%etalact%'"),
            ),
            ("Pacientes com exame pendente",
             um("SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE status='pendente'")),
            ("Pacientes com valor crítico",
             um("SELECT COUNT(DISTINCT paciente_id) FROM exames WHERE critico=1")),
            ("Gestantes", um("SELECT COUNT(*) FROM pacientes WHERE gestante=1")),
        ]
    finally:
        conexao.close()

    linhas = ["| Tabela | Registros |", "| --- | ---: |"]
    linhas += [f"| {nome} | {valor:,} |" for nome, valor in indicadores]
    linhas += [
        "",
        "Casos que as regras de segurança precisam exercitar — se não existissem na base, "
        "as regras nunca disparariam e passariam a impressão falsa de estarem funcionando:",
        "",
        "| Caso | Pacientes |",
        "| --- | ---: |",
    ]
    linhas += [f"| {nome} | {valor} |" for nome, valor in casos]
    return "\n".join(linhas)


def bloco_indice(cfg) -> str:
    dados = _ler_json(cfg.dir_indices / "faiss_medgraph" / "estatisticas.json")
    if not dados:
        return AUSENTE.format(comando="make indexar")

    linhas = ["| Indicador | Valor |", "| --- | ---: |",
              f"| Trechos indexados | {dados['total_trechos']:,} |"]
    for fonte, quantidade in dados.get("trechos_por_fonte", {}).items():
        linhas.append(f"| &nbsp;&nbsp;· {fonte} | {quantidade:,} |")
    linhas += [
        f"| Documentos originais | {dados['documentos_originais']:,} |",
        f"| Caracteres indexados | {dados['caracteres_totais']:,} |",
        f"| Modelo de embedding | `{dados['modelo_embedding']}` |",
        f"| Dimensão do vetor | {dados['dimensao']} |",
        f"| Tempo de construção | {dados['duracao_s']:.1f} s |",
    ]
    return "\n".join(linhas)


def bloco_avaliacao(cfg) -> str:
    dados = _ler_json(cfg.dir_docs / "avaliacao_resultados.json")
    if not dados:
        return AUSENTE.format(comando="make avaliar")

    conjunto = dados["conjunto_de_teste"]
    linhas = [
        f"Conjunto de teste: **{conjunto['casos_avaliados']} casos** de "
        f"{conjunto['casos_disponiveis']} disponíveis · distribuição "
        f"`{conjunto['distribuicao']}`",
        "",
        "| Sistema | N | Accuracy | Macro-F1 | F1 yes | F1 no | F1 maybe | Formato | Latência |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in dados["sistemas"]:
        c = s["por_classe"]
        linhas.append(
            f"| {s['sistema']} | {s['total']} | {s['accuracy']:.3f} | **{s['macro_f1']:.3f}** "
            f"| {c['yes']['f1']:.3f} | {c['no']['f1']:.3f} | {c['maybe']['f1']:.3f} "
            f"| {s['taxa_adesao_formato']:.0%} | {s['latencia_media_ms']:,.0f} ms |"
        )

    referencia = dados["referencia_externa"]
    linhas += [
        "",
        f"Referência externa: especialistas humanos alcançam "
        f"**{referencia['especialista_humano']:.0%}** neste mesmo conjunto "
        f"({referencia['fonte']}).",
        "",
        "#### Distribuição das previsões",
        "",
        "Delata o modelo que colapsou numa única classe — o modo de falha mais comum e o "
        "mais fácil de confundir com bom desempenho.",
        "",
        "| Sistema | yes | no | maybe | sem rótulo |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in dados["sistemas"]:
        d = s["distribuicao_previsoes"]
        linhas.append(
            f"| {s['sistema']} | {d.get('yes', 0)} | {d.get('no', 0)} "
            f"| {d.get('maybe', 0)} | {d.get('(sem rótulo)', 0)} |"
        )

    linhas += ["", "#### Custo desta avaliação", "",
               "| Modelo | Chamadas | Tokens entrada | Tokens saída | Custo |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for modelo, uso in sorted(dados["custo"].items()):
        linhas.append(
            f"| `{modelo}` | {uso['chamadas']:,} | {uso['tokens_entrada']:,} "
            f"| {uso['tokens_saida']:,} | US$ {uso['custo_usd']:.6f} |"
        )
    linhas.append(f"| **Total** | | | | **US$ {dados['custo_total_usd']:.6f}** |")
    linhas += [
        "",
        "O modelo local aparece com custo zero e o volume processado registrado — é o que "
        "permite comparar o custo por consulta entre o modelo servido localmente e a API paga.",
    ]
    return "\n".join(linhas)


def bloco_treino(cfg) -> str:
    cartoes = list(cfg.dir_adapters.glob("*/cartao_de_treino.json"))
    if not cartoes:
        return (
            "_O fine-tuning é executado no Google Colab (`notebooks/colab/"
            "01_finetune_qlora_pubmedqa.ipynb`), porque exige GPU com CUDA. "
            "Assim que o notebook for executado e o adapter descompactado em "
            "`models/adapters/`, esta seção passa a exibir os hiperparâmetros "
            "efetivos, as versões das bibliotecas, a duração e a curva de perda "
            "registrados no cartão de treino._"
        )

    dados = json.loads(cartoes[0].read_text(encoding="utf-8"))
    linhas = [
        "| Item | Valor |", "| --- | --- |",
        f"| Modelo base | `{dados['modelo_base']}` |",
        f"| Método | {dados['metodo']} |",
        f"| GPU | {dados['gpu']['nome']} ({dados['gpu']['memoria_gb']} GB) |",
        f"| Exemplos de treino | {dados['dados']['treino']:,} |",
        f"| Duração | {dados['duracao_legivel']} |",
        f"| Perda de treino | {dados['perda_treino_inicial']} → {dados['perda_treino_final']} |",
        f"| Perda de validação | {dados['perda_validacao_inicial']} → {dados['perda_validacao_final']} |",
        "",
        "| Hiperparâmetro | Valor |", "| --- | --- |",
    ]
    for chave, valor in dados["hiperparametros"].items():
        linhas.append(f"| `{chave}` | `{valor}` |")
    return "\n".join(linhas)


def bloco_testes() -> str:
    import subprocess

    resultado = subprocess.run(
        [str(RAIZ / ".venv/bin/python"), "-m", "pytest", "tests/", "-q", "--no-header"],
        capture_output=True, text=True, cwd=str(RAIZ),
        env={"PYTHONPATH": f"{RAIZ}:{RAIZ / 'src'}", "PATH": "/usr/bin:/bin"},
    )
    ultima = [linha for linha in resultado.stdout.splitlines() if "passed" in linha or "failed" in linha]
    return f"```\n{ultima[-1] if ultima else '(não foi possível executar a suíte)'}\n```"


def bloco_rastreabilidade(cfg) -> str:
    caminho = cfg.dir_docs / "rastreabilidade.md"
    if not caminho.exists():
        return AUSENTE.format(comando="make rastreabilidade")
    texto = caminho.read_text(encoding="utf-8")
    inicio = texto.find("| Requisitos no catálogo")
    fim = texto.find("### Requisitos ainda sem cobertura")
    if inicio == -1:
        return "_(não foi possível extrair o resumo de cobertura)_"
    return texto[inicio : fim if fim > 0 else inicio + 400].strip()


# =============================================================================
# MONTAGEM
# =============================================================================
def main() -> int:
    cfg = obter_settings()
    base = cfg.dir_docs / "relatorio_base.md"

    if not base.exists():
        print(f"ERRO: {base} não encontrado. É a narrativa do relatório, escrita à mão.")
        return 1

    texto = base.read_text(encoding="utf-8")

    substituicoes = {
        "{{DATA}}": date.today().isoformat(),
        "{{DADOS}}": bloco_dados(cfg),
        "{{PRONTUARIOS}}": bloco_prontuarios(cfg),
        "{{INDICE}}": bloco_indice(cfg),
        "{{TREINO}}": bloco_treino(cfg),
        "{{AVALIACAO}}": bloco_avaliacao(cfg),
        "{{TESTES}}": bloco_testes(),
        "{{RASTREABILIDADE}}": bloco_rastreabilidade(cfg),
    }

    faltando = [chave for chave in substituicoes if chave not in texto]
    for chave, valor in substituicoes.items():
        texto = texto.replace(chave, valor)

    # Um marcador que sobrou no texto final é erro de digitação no template, e
    # apareceria como "{{ALGO}}" no meio do relatório entregue.
    import re

    sobraram = re.findall(r"\{\{[A-Z_]+\}\}", texto)
    if sobraram:
        print(f"ATENÇÃO: marcadores não substituídos no relatório: {set(sobraram)}")

    destino = cfg.dir_docs / "relatorio_tecnico.md"
    destino.write_text(texto, encoding="utf-8")

    print(f"Relatório gerado em {destino.relative_to(RAIZ)}")
    print(f"  {len(texto.splitlines())} linhas | {len(texto):,} caracteres")
    if faltando:
        print(f"  marcadores não usados pelo template: {faltando}")
    return 1 if sobraram else 0


if __name__ == "__main__":
    raise SystemExit(main())
