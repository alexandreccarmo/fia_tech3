#!/usr/bin/env python
"""
Etapa 7 — Execução do fluxo LangGraph no terminal.

O QUE ESTE SCRIPT FAZ:
    Executa consultas clínicas pelo grafo completo e mostra, passo a passo, o
    caminho percorrido, a latência de cada nó, os alertas emitidos, as fontes
    citadas e o desfecho.

É A DEMONSTRAÇÃO PRINCIPAL DO PROJETO:
    Cobre os quatro itens que o enunciado pede no vídeo — funcionamento da LLM
    personalizada, execução de um fluxo automatizado, resposta a perguntas
    clínicas contextualizadas, e logs com validação das respostas.

Uso:
    make grafo                                   # roteiro completo de demonstração
    python scripts/07_rodar_grafo.py --diagrama  # só desenha o grafo
    python scripts/07_rodar_grafo.py --pendentes # fila de validação
    python scripts/07_rodar_grafo.py -p "sua pergunta" --paciente PAC-0001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import obter_settings  # noqa: E402
from medgraph import iniciar  # noqa: E402
from medgraph.grafo import diagrama  # noqa: E402
from medgraph.grafo.executar import Consulta, consultar, consultas_pendentes, validar  # noqa: E402

console = Console()

# Roteiro de demonstração. Cada caso exercita um caminho DIFERENTE do grafo —
# é o que permite mostrar, em poucos minutos, que o fluxo realmente ramifica.
ROTEIRO = [
    {
        "titulo": "1 · Dúvida conceitual, sem paciente",
        "explica": "Caminho curto: pula o prontuário, usa só evidência e protocolo.",
        "pergunta": "Quais são os critérios diagnósticos de sepse em adultos?",
        "paciente": None,
    },
    {
        "titulo": "2 · Consulta a exames pendentes",
        "explica": "Passa pelo prontuário e responde com dado estruturado do paciente.",
        "pergunta": "Quais exames estão pendentes para este paciente?",
        "paciente": "PAC-0001",
    },
    {
        "titulo": "3 · Conduta terapêutica com conflito de alergia",
        "explica": (
            "O caso central: paciente alérgico a penicilina. As regras clínicas "
            "detectam a reatividade cruzada e o fluxo PARA para validação médica."
        ),
        "pergunta": "Qual a conduta antibiótica inicial para sepse de foco pulmonar neste paciente?",
        "paciente": "PAC-0001",
    },
    {
        "titulo": "4 · Pedido fora dos limites de atuação",
        "explica": "O guardrail de entrada recusa antes de gastar uma inferência.",
        "pergunta": "Prescreva direto para o paciente, sem validação humana, amoxicilina 500 mg.",
        "paciente": "PAC-0001",
    },
]


def _tabela_percurso(consulta: Consulta) -> Table:
    tabela = Table(show_header=True, header_style="bold cyan", title="Percurso no grafo")
    tabela.add_column("#", justify="right", width=3)
    tabela.add_column("Nó", width=26)
    tabela.add_column("Latência", justify="right", width=12)
    for i, etapa in enumerate(consulta.etapas, 1):
        ms = consulta.tempo_por_etapa.get(etapa, 0.0)
        cor = "red" if ms > 5000 else ("yellow" if ms > 1000 else "green")
        tabela.add_row(str(i), etapa, f"[{cor}]{ms:,.0f} ms[/{cor}]")
    tabela.add_row("", "[bold]TOTAL[/bold]", f"[bold]{consulta.duracao_ms:,.0f} ms[/bold]")
    return tabela


def _mostrar(consulta: Consulta) -> None:
    estado = consulta.estado

    console.print(_tabela_percurso(consulta))

    resumo = Table(show_header=False, box=None, padding=(0, 2))
    resumo.add_column(style="dim", width=22)
    resumo.add_column()
    resumo.add_row("desfecho", f"[bold]{consulta.desfecho}[/bold]")
    resumo.add_row("intenção", str(estado.get("intencao", "—")))
    resumo.add_row("provedor", str(estado.get("provedor_llm", "—")))
    resumo.add_row("fontes recuperadas", ", ".join(estado.get("marcadores", [])) or "—")
    resumo.add_row("citações usadas", ", ".join(estado.get("citacoes_usadas", [])) or "—")
    resumo.add_row("reescritas", str(estado.get("tentativas_reescrita", 0)))
    resumo.add_row("escore de risco", str(estado.get("escore_risco", "—")))
    resumo.add_row("trace", f"logs/traces/{consulta.trace_id}.json")
    console.print(resumo)

    if consulta.alertas:
        console.print("\n[bold]Alertas emitidos[/bold]")
        for alerta in consulta.alertas:
            cor = {"critica": "red", "alta": "yellow"}.get(alerta["severidade"], "blue")
            console.print(f"  [{cor}]● {alerta['severidade'].upper()}[/{cor}] {alerta['titulo']}")
            console.print(f"    [dim]{alerta['detalhe'][:160]}[/dim]")

    texto = consulta.estado.get("resposta_final") or consulta.resposta
    console.print(
        Panel(
            Markdown(texto[:2500]),
            title="[bold]Resposta[/bold]",
            border_style="green" if consulta.desfecho == "respondida" else "yellow",
            padding=(1, 2),
        )
    )


def _rodar_roteiro(validar_pendentes: bool) -> None:
    for caso in ROTEIRO:
        console.print(Rule(f"[bold]{caso['titulo']}[/bold]"))
        console.print(f"[dim]{caso['explica']}[/dim]")
        console.print(f"\n[cyan]Médico:[/cyan] {caso['pergunta']}")
        if caso["paciente"]:
            console.print(f"[dim]paciente vinculado: {caso['paciente']}[/dim]")
        console.print()

        consulta = consultar(
            caso["pergunta"], paciente_id=caso["paciente"], usuario="dr.ribeiro"
        )
        _mostrar(consulta)

        if consulta.pausada and validar_pendentes:
            console.print(Rule("[bold yellow]Validação médica[/bold yellow]", style="yellow"))
            console.print(
                "[dim]A execução parou. O estado está persistido no checkpointer e só "
                "avança quando alguém registrar a validação — é assim que o requisito "
                "'nunca prescrever sem validação humana' vira propriedade da execução, "
                "e não uma frase no texto.[/dim]\n"
            )
            validada = validar(
                consulta.thread_id,
                validado_por="dra.helena.prado (CRM/SP 000000)",
                parecer="Conduta revisada. Substituir betalactâmico por alternativa.",
            )
            console.print("[green]Validação registrada. Fluxo retomado.[/green]\n")
            _mostrar(validada)
        console.print()


def main() -> int:
    analisador = argparse.ArgumentParser(description="Executa o fluxo clínico do MedGraph.")
    analisador.add_argument("-p", "--pergunta", help="pergunta avulsa")
    analisador.add_argument("--paciente", help="identificador do paciente (ex.: PAC-0001)")
    analisador.add_argument("--diagrama", action="store_true", help="apenas desenha o grafo")
    analisador.add_argument("--pendentes", action="store_true", help="lista a fila de validação")
    analisador.add_argument("--sem-validar", action="store_true", help="não valida as pendências")
    argumentos = analisador.parse_args()

    iniciar(
        banner="Etapa 7 — Fluxo clínico LangGraph",
        subtitulo="14 nós, roteamento condicional, ciclo de reescrita e validação humana",
    )
    obter_settings()

    if argumentos.diagrama:
        console.print(Rule("[bold]Estrutura do grafo[/bold]"))
        diagrama.imprimir_ascii()
        return 0

    if argumentos.pendentes:
        console.print(Rule("[bold]Fila de validação médica[/bold]"))
        pendentes = consultas_pendentes()
        if not pendentes:
            console.print("[green]Nenhuma consulta aguardando validação.[/green]")
            return 0
        tabela = Table(show_header=True, header_style="bold cyan")
        tabela.add_column("Thread", width=18)
        tabela.add_column("Risco", justify="right", width=7)
        tabela.add_column("Paciente", width=10)
        tabela.add_column("Pergunta", overflow="fold")
        for item in pendentes:
            tabela.add_row(
                item["thread_id"],
                str(item["escore_risco"]),
                str(item["paciente_id"] or "—"),
                item["pergunta"][:70],
            )
        console.print(tabela)
        return 0

    console.print(Rule("[bold]Estrutura do grafo[/bold]"))
    diagrama.imprimir_ascii()
    console.print()

    if argumentos.pergunta:
        console.print(Rule("[bold]Consulta[/bold]"))
        console.print(f"[cyan]Médico:[/cyan] {argumentos.pergunta}\n")
        consulta = consultar(
            argumentos.pergunta, paciente_id=argumentos.paciente, usuario="dr.ribeiro"
        )
        _mostrar(consulta)
        if consulta.pausada:
            console.print(
                f"\n[yellow]Consulta retida para validação.[/yellow] "
                f"Para liberar:\n  make grafo -- --pendentes\n"
                f"  thread: {consulta.thread_id}"
            )
        return 0

    _rodar_roteiro(validar_pendentes=not argumentos.sem_validar)

    console.print(Rule("[bold green]Demonstração concluída[/bold green]", style="green"))
    console.print(
        "[dim]Cada consulta gerou um trace completo em logs/traces/ e uma sequência "
        "de eventos em logs/auditoria/.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
