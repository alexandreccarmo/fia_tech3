#!/usr/bin/env python
"""
Demonstracao visual da fundacao do projeto (Etapa 0).

O QUE FAZ:
    Simula uma consulta clinica passando por quatro "nos" ficticios e mostra,
    ao vivo, o que a infraestrutura da Etapa 0 produz:

      - o banner de identificacao da execucao;
      - o log colorido, evento a evento, no console;
      - a trilha de auditoria em JSONL;
      - o arquivo de trace completo da consulta;
      - a contabilizacao de tokens e custo;
      - a trava de orcamento entrando em acao.

POR QUE ESTE SCRIPT EXISTE:
    A Etapa 0 nao entrega funcionalidade visivel - entrega infraestrutura.
    Sem uma demonstracao, "logging detalhado para rastreamento e auditoria"
    (item 3 do enunciado) fica sendo uma promessa. Este script transforma a
    promessa em algo que da para assistir, e serve de trecho de abertura para
    o video de entrega.

    Os nos aqui sao propositalmente falsos: eles apenas dormem alguns
    milissegundos. O que esta sendo demonstrado e a INSTRUMENTACAO, nao a
    logica clinica - essa chega nas Etapas 5 a 7, e vai usar exatamente estes
    mesmos mecanismos.

Uso:
    python scripts/demo_fundacao.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.auditoria import (  # noqa: E402
    Desfecho,
    TipoEvento,
    abrir_trilha,
    instrumentar,
    registrar,
)
from medgraph.llm.custo import OrcamentoExcedidoError, contador, reiniciar_contador  # noqa: E402

console = Console()


# =============================================================================
# NOS SIMULADOS
# =============================================================================
# Cada funcao abaixo imita um no do grafo que sera construido na Etapa 7.
# O decorator @instrumentar e o mesmo que os nos reais vao usar: ele registra
# inicio, fim, duracao, chaves do estado e delta produzido, sem exigir uma
# unica linha de log dentro da funcao.
# =============================================================================


@instrumentar("guardrail_entrada", tipo=TipoEvento.GUARDRAIL)
def guardrail_entrada(estado: dict) -> dict:
    """Simula a verificacao de escopo e a anonimizacao da pergunta. [REQ-3a]"""
    time.sleep(0.04)
    registrar(
        TipoEvento.ANONIMIZACAO,
        "1 identificador removido da pergunta",
        etapa="guardrail_entrada",
        tipos_removidos=["nome_paciente"],
    )
    return {"aprovado_entrada": True, "pergunta_limpa": estado["pergunta"]}


@instrumentar("consultar_prontuario", tipo=TipoEvento.BANCO)
def consultar_prontuario(estado: dict) -> dict:
    """Simula a leitura da base estruturada de pacientes. [REQ-2a]"""
    time.sleep(0.07)
    return {
        "paciente": {"id": estado["paciente_id"], "idade": 67, "alergias": ["penicilina"]},
        "exames_pendentes": ["hemocultura", "lactato"],
    }


@instrumentar("recuperar_evidencia", tipo=TipoEvento.RECUPERACAO)
def recuperar_evidencia(estado: dict) -> dict:
    """Simula a busca no indice vetorial, devolvendo as fontes. [REQ-3c]"""
    time.sleep(0.09)
    return {
        "fontes": ["E1:pubmed_28123456", "P2:protocolo_sepse_v3", "C1:prontuario_P0042"],
        "trechos_recuperados": 3,
    }


@instrumentar("raciocinio_clinico", tipo=TipoEvento.LLM)
def raciocinio_clinico(estado: dict) -> dict:
    """Simula a inferencia da LLM fine-tunada e contabiliza o consumo."""
    time.sleep(0.12)

    # Numeros ficticios, apenas para exercitar a contabilidade.
    contador().registrar_uso(
        modelo="medgraph",
        tokens_entrada=1420,
        tokens_saida=310,
        origem="demo.raciocinio_clinico",
    )
    return {
        "resposta": (
            "Iniciar antibioticoterapia empirica conforme o protocolo interno [P2]. "
            "Atencao: paciente com alergia a penicilina registrada [C1]."
        )
    }


@instrumentar("guardrail_saida", tipo=TipoEvento.GUARDRAIL)
def guardrail_saida(estado: dict) -> dict:
    """Simula a validacao da resposta antes da entrega. [REQ-3a][REQ-3c]"""
    time.sleep(0.03)
    tem_citacao = "[" in estado["resposta"]
    registrar(
        TipoEvento.REGRA_CLINICA,
        "Alergia a penicilina conflita com a classe sugerida",
        etapa="guardrail_saida",
        gatilho="alergia_conhecida_do_paciente",
        peso=0.90,
    )
    registrar(
        TipoEvento.ALERTA,
        "Alerta emitido para a equipe medica",
        etapa="guardrail_saida",
        severidade="alta",
    )
    return {"aprovado_saida": tem_citacao, "risco": 0.90}


# =============================================================================
# DEMONSTRACAO
# =============================================================================
def main() -> int:
    iniciar(
        banner="Etapa 0 - Demonstracao da fundacao",
        subtitulo="logging em tres destinos, trilha de auditoria e controle de custo",
    )
    cfg = obter_settings()
    reiniciar_contador(limite_usd=cfg.max_custo_usd_sessao)

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]1. Execucao instrumentada de uma consulta[/bold]"))
    # -------------------------------------------------------------------------
    estado = {
        "pergunta": "Qual a conduta inicial para suspeita de sepse na paciente Maria Silva?",
        "paciente_id": "P0042",
    }

    with abrir_trilha(
        pergunta=estado["pergunta"],
        usuario="dr.ribeiro",
        paciente_id=estado["paciente_id"],
    ) as trilha:
        estado |= guardrail_entrada(estado)
        estado |= consultar_prontuario(estado)
        estado |= recuperar_evidencia(estado)
        estado |= raciocinio_clinico(estado)
        estado |= guardrail_saida(estado)

        # Risco acima do limiar: a consulta nao e entregue, fica pendente de
        # validacao humana. E o comportamento exigido pelo item 3 do enunciado.
        if estado["risco"] >= cfg.limiar_risco_validacao_humana:
            registrar(
                TipoEvento.VALIDACAO_HUMANA,
                "Risco acima do limiar - resposta retida para validacao medica",
                risco=estado["risco"],
                limiar=cfg.limiar_risco_validacao_humana,
            )
            trilha.desfecho = Desfecho.AGUARDANDO_VALIDACAO

        trace_id = trilha.trace_id
        tempos = trilha.tempo_por_etapa()
        etapas = trilha.etapas_executadas()

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]2. Trilha do grafo (o que o painel visual vai desenhar)[/bold]"))
    # -------------------------------------------------------------------------
    tabela = Table(show_header=True, header_style="bold cyan")
    tabela.add_column("#", justify="right", width=3)
    tabela.add_column("Etapa executada", width=26)
    tabela.add_column("Latencia", justify="right", width=12)
    for i, etapa_nome in enumerate(etapas, 1):
        tabela.add_row(str(i), etapa_nome, f"{tempos.get(etapa_nome, 0):.1f} ms")
    tabela.add_row("", "[bold]TOTAL[/bold]", f"[bold]{sum(tempos.values()):.1f} ms[/bold]")
    console.print(tabela)

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]3. Trilha de auditoria em JSONL (3 primeiras linhas)[/bold]"))
    # -------------------------------------------------------------------------
    from medgraph.logging_config import caminho_auditoria_do_dia

    arquivo_jsonl = caminho_auditoria_do_dia(cfg)
    console.print(f"[dim]{arquivo_jsonl}[/dim]\n")
    linhas = arquivo_jsonl.read_text(encoding="utf-8").splitlines()
    for linha in linhas[-len(etapas) - 2 :][:3]:
        console.print(
            Syntax(json.dumps(json.loads(linha), ensure_ascii=False), "json", word_wrap=True)
        )

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]4. Trace completo da consulta[/bold]"))
    # -------------------------------------------------------------------------
    arquivo_trace = cfg.dir_traces / f"{trace_id}.json"
    dossie = json.loads(arquivo_trace.read_text(encoding="utf-8"))
    console.print(f"[dim]{arquivo_trace}[/dim]\n")
    console.print(f"  trace_id .......... [cyan]{dossie['trace_id']}[/cyan]")
    console.print(f"  usuario ........... {dossie['usuario']}")
    console.print(f"  paciente .......... {dossie['paciente_id']}")
    console.print(f"  desfecho .......... [yellow]{dossie['desfecho']}[/yellow]")
    console.print(f"  duracao total ..... {dossie['duracao_total_ms']:.1f} ms")
    console.print(f"  eventos ........... {dossie['total_eventos']}")
    console.print(f"  etapas ............ {' -> '.join(dossie['etapas_executadas'])}")

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]5. Contabilidade de consumo[/bold]"))
    # -------------------------------------------------------------------------
    console.print(contador().tabela_resumo())

    # Persiste o consumo em logs/custos.jsonl. Arquivo separado da trilha de
    # auditoria porque acumula ao longo de todo o projeto e alimenta a tabela
    # de custo do relatorio tecnico.
    contador().salvar(cfg)
    console.print(f"\n[dim]consumo persistido em {cfg.dir_logs / 'custos.jsonl'}[/dim]")

    # -------------------------------------------------------------------------
    console.print(Rule("[bold]6. Trava de orcamento em acao[/bold]"))
    # -------------------------------------------------------------------------
    c = reiniciar_contador(limite_usd=0.01)
    c.registrar_uso("gpt-4o-mini", tokens_entrada=60_000, tokens_saida=0, origem="demo")
    console.print(f"  gasto acumulado: US$ {c.total_usd:.4f} | teto: US$ {c.limite_usd:.2f}")
    try:
        c.verificar_orcamento(custo_previsto_usd=0.05)
        console.print("  [red]a trava NAO funcionou[/red]")
        return 1
    except OrcamentoExcedidoError as exc:
        console.print("  [green]chamada bloqueada como esperado:[/green]")
        for linha in str(exc).splitlines():
            console.print(f"    [dim]{linha}[/dim]")

    console.print(Rule(style="green"))
    console.print(
        "\n[bold green]Fundacao validada.[/bold green] "
        "Todos os mecanismos das Etapas 1 a 8 vao se apoiar nesta infraestrutura.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
