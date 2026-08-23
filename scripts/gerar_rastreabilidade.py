#!/usr/bin/env python
"""
[REQ-4] Gerador da matriz de rastreabilidade requisito -> codigo.

O QUE FAZ:
    Varre o repositorio procurando as tags [REQ-xx] usadas nas docstrings e
    comentarios, cruza com o catalogo em medgraph/requisitos.py e escreve
    docs/rastreabilidade.md - uma tabela que liga cada exigencia do enunciado
    do Tech Challenge ao arquivo e a linha exatos onde ela foi atendida.

POR QUE EXISTE:
    Um projeto deste tamanho tem dezenas de arquivos. Provar que os treze
    itens cobrados no PDF foram todos implementados, lendo o codigo de ponta
    a ponta, e trabalhoso para quem corrige. Este documento resolve isso: o
    professor abre uma pagina e ve, item por item, onde procurar.

    Gerar automaticamente (em vez de manter a tabela na mao) tem duas
    vantagens: nunca fica desatualizada, e denuncia requisito sem cobertura -
    se algum [REQ-xx] nao aparecer em lugar nenhum do codigo, o relatorio
    marca em destaque.

Uso:
    make rastreabilidade
    python scripts/gerar_rastreabilidade.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))

from medgraph.requisitos import CATALOGO, POR_CODIGO, listar_categorias  # noqa: E402

# Onde procurar as tags. Deliberadamente exclui .venv, data/ e logs/.
PASTAS_VARRIDAS = ("src", "config", "tests", "scripts", "notebooks")
EXTENSOES = {".py", ".yaml", ".yml", ".ipynb", ".md", ".sh"}

# Uma tag e sempre [REQ-] seguido de digitos e, opcionalmente, uma letra.
PADRAO_TAG = re.compile(r"\[(REQ-[0-9]+[a-zA-Z]?)\]")


@dataclass(frozen=True)
class Ocorrencia:
    codigo: str
    arquivo: str
    linha: int
    contexto: str


def varrer() -> tuple[list[Ocorrencia], set[str]]:
    """
    Percorre o repositorio e coleta todas as tags encontradas.

    Devolve as ocorrencias validas e o conjunto de codigos desconhecidos -
    tags que existem no codigo mas nao no catalogo, quase sempre erro de
    digitacao.
    """
    ocorrencias: list[Ocorrencia] = []
    desconhecidos: set[str] = set()

    for pasta in PASTAS_VARRIDAS:
        base = RAIZ / pasta
        if not base.is_dir():
            continue
        for arquivo in sorted(base.rglob("*")):
            if not arquivo.is_file() or arquivo.suffix not in EXTENSOES:
                continue
            if any(parte in {".venv", "__pycache__", ".ipynb_checkpoints"} for parte in arquivo.parts):
                continue

            try:
                linhas = arquivo.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue

            for numero, texto in enumerate(linhas, start=1):
                for codigo in PADRAO_TAG.findall(texto):
                    if codigo not in POR_CODIGO:
                        desconhecidos.add(f"{codigo} ({arquivo.relative_to(RAIZ)}:{numero})")
                        continue
                    ocorrencias.append(
                        Ocorrencia(
                            codigo=codigo,
                            arquivo=str(arquivo.relative_to(RAIZ)),
                            linha=numero,
                            contexto=texto.strip().lstrip("#").strip()[:110],
                        )
                    )

    return ocorrencias, desconhecidos


def montar_documento(ocorrencias: list[Ocorrencia], desconhecidos: set[str]) -> str:
    por_codigo: dict[str, list[Ocorrencia]] = defaultdict(list)
    for o in ocorrencias:
        por_codigo[o.codigo].append(o)

    cobertos = [r for r in CATALOGO if por_codigo.get(r.codigo)]
    descobertos = [r for r in CATALOGO if not por_codigo.get(r.codigo)]

    linhas: list[str] = [
        "# Matriz de rastreabilidade",
        "",
        "> **Documento gerado automaticamente.** Não edite à mão — rode `make rastreabilidade`.",
        f"> Última geração: {date.today().isoformat()}",
        "",
        "Este documento liga cada exigência do enunciado do Tech Challenge (Fase 3 — 8IADT)",
        "ao ponto exato do código onde ela é atendida. As referências vêm das tags",
        "`[REQ-xx]` escritas nas docstrings do projeto.",
        "",
        "## Resumo da cobertura",
        "",
        f"| Requisitos no catálogo | {len(CATALOGO)} |",
        "| --- | --- |",
        f"| **Com implementação identificada** | **{len(cobertos)}** |",
        f"| Ainda sem implementação | {len(descobertos)} |",
        f"| Total de referências no código | {len(ocorrencias)} |",
        "",
    ]

    if descobertos:
        linhas += [
            "### Requisitos ainda sem cobertura",
            "",
            "São itens previstos para etapas seguintes do projeto.",
            "",
            "| Código | Categoria | Descrição |",
            "| --- | --- | --- |",
        ]
        linhas += [
            f"| `{r.codigo}` | {r.categoria} | {r.descricao} |" for r in descobertos
        ]
        linhas.append("")

    if desconhecidos:
        linhas += [
            "### ⚠️ Tags inválidas encontradas",
            "",
            "As referências abaixo não existem no catálogo — provável erro de digitação.",
            "",
        ]
        linhas += [f"- `{d}`" for d in sorted(desconhecidos)]
        linhas.append("")

    linhas += ["## Detalhamento por requisito", ""]

    for categoria in listar_categorias():
        requisitos_da_categoria = [r for r in CATALOGO if r.categoria == categoria]
        linhas += [f"### {categoria}", ""]

        for requisito in requisitos_da_categoria:
            achados = sorted(por_codigo.get(requisito.codigo, []), key=lambda o: (o.arquivo, o.linha))
            marca = "✅" if achados else "⏳"

            linhas += [
                f"#### {marca} `{requisito.codigo}` — {requisito.descricao}",
                "",
                f"*Origem no PDF: {requisito.origem}*",
                "",
            ]

            if not achados:
                linhas += ["_Sem implementação identificada até o momento._", ""]
                continue

            linhas += ["| Arquivo | Linha | Contexto |", "| --- | ---: | --- |"]
            linhas += [
                f"| `{o.arquivo}` | {o.linha} | {o.contexto.replace('|', '\\|')} |"
                for o in achados
            ]
            linhas.append("")

    return "\n".join(linhas) + "\n"


def main() -> int:
    ocorrencias, desconhecidos = varrer()
    documento = montar_documento(ocorrencias, desconhecidos)

    destino = RAIZ / "docs" / "rastreabilidade.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(documento, encoding="utf-8")

    cobertos = {o.codigo for o in ocorrencias}
    print(f"Matriz gerada em {destino.relative_to(RAIZ)}")
    print(f"  {len(cobertos)}/{len(CATALOGO)} requisitos com implementacao identificada")
    print(f"  {len(ocorrencias)} referencias [REQ-xx] encontradas no codigo")

    if desconhecidos:
        print("\n  ATENCAO - tags invalidas (provavel erro de digitacao):")
        for d in sorted(desconhecidos):
            print(f"    - {d}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
