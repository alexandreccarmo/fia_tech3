"""
[REQ-4][REQ-E3] Gera docs/MedGraph-Guia-do-Projeto.pdf a partir de Markdown.

POR QUE UM GERADOR, E NAO UM PDF COMMITADO DIRETO:
    Um PDF binario no repositorio nao tem diff, nao tem revisao e envelhece sem
    aviso - o mesmo problema que levou o relatorio tecnico a ser gerado a partir
    de docs/relatorio_base.md em vez de escrito a mao.

    Aqui a fonte e Markdown versionado, e o PDF e artefato.

POR QUE A DIRETIVA DE INCLUSAO:
    O passo a passo de execucao precisa existir como .md legivel no repositorio
    E dentro do PDF. Manter as duas copias na mao garantiria que uma delas
    ficaria para tras. A diretiva

        {{incluir: execucao_passo_a_passo.md}}

    resolve o arquivo no momento da geracao, entao ha uma fonte so.

Rodar com:  make guia
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

RAIZ = Path(__file__).resolve().parent.parent
FONTE = RAIZ / "docs" / "guia_do_projeto.md"
SAIDA = RAIZ / "docs" / "MedGraph-Guia-do-Projeto.pdf"

# As fontes do sistema sao usadas porque as embutidas do reportlab nao cobrem
# travessao nem aspas curvas - caracteres que o texto em portugues usa o tempo
# todo e que sairiam como retangulos pretos.
FONTES_MAC = "/System/Library/Fonts/Supplemental"
FAMILIA = [
    ("Corpo", f"{FONTES_MAC}/Arial.ttf"),
    ("Corpo-Bold", f"{FONTES_MAC}/Arial Bold.ttf"),
    ("Corpo-Italic", f"{FONTES_MAC}/Arial Italic.ttf"),
    ("Mono", f"{FONTES_MAC}/Courier New.ttf"),
]

TINTA = colors.HexColor("#1a1a1a")
SUAVE = colors.HexColor("#5b6470")
AZUL = colors.HexColor("#1f4e79")
LINHA = colors.HexColor("#d5dae1")
FUNDO = colors.HexColor("#f4f6f8")

LARGURA_UTIL = 16.4 * cm


def registrar_fontes() -> bool:
    """Devolve False quando as fontes do sistema nao estao onde esperamos."""
    try:
        for nome, arquivo in FAMILIA:
            pdfmetrics.registerFont(TTFont(nome, arquivo))
    except Exception:
        return False
    pdfmetrics.registerFontFamily(
        "Corpo", "Corpo", "Corpo-Bold", "Corpo-Italic", "Corpo-Bold"
    )
    return True


def montar_estilos() -> dict[str, ParagraphStyle]:
    padrao = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("titulo", parent=padrao["Title"], fontName="Corpo-Bold",
                                 fontSize=30, leading=35, textColor=AZUL, spaceAfter=6),
        "subtitulo": ParagraphStyle("subtitulo", parent=padrao["Normal"], fontName="Corpo",
                                    fontSize=15, leading=20, textColor=SUAVE,
                                    alignment=1, spaceAfter=18),
        "capa_nota": ParagraphStyle("capa_nota", parent=padrao["Normal"], fontName="Corpo",
                                    fontSize=10.5, leading=16, textColor=SUAVE,
                                    alignment=1, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=padrao["Heading1"], fontName="Corpo-Bold",
                             fontSize=18, leading=23, textColor=AZUL,
                             spaceBefore=20, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=padrao["Heading2"], fontName="Corpo-Bold",
                             fontSize=13, leading=17, textColor=TINTA,
                             spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=padrao["Heading3"], fontName="Corpo-Bold",
                             fontSize=11, leading=15, textColor=TINTA,
                             spaceBefore=10, spaceAfter=4),
        "corpo": ParagraphStyle("corpo", parent=padrao["Normal"], fontName="Corpo",
                                fontSize=10, leading=15.5, textColor=TINTA,
                                alignment=TA_JUSTIFY, spaceAfter=8),
        "citacao": ParagraphStyle("citacao", parent=padrao["Normal"], fontName="Corpo-Italic",
                                  fontSize=10.5, leading=16, textColor=AZUL,
                                  leftIndent=14, rightIndent=14, spaceBefore=4,
                                  spaceAfter=10),
        "item": ParagraphStyle("item", parent=padrao["Normal"], fontName="Corpo",
                               fontSize=10, leading=15, textColor=TINTA,
                               leftIndent=16, bulletIndent=5, spaceAfter=4),
        # parent=Code traz leftIndent=36, que empurraria todo bloco para dentro.
        "codigo": ParagraphStyle("codigo", parent=padrao["Code"], fontName="Mono",
                                 fontSize=7.9, leading=10.6, textColor=TINTA,
                                 leftIndent=0, rightIndent=0, spaceBefore=0, spaceAfter=0),
        "celula": ParagraphStyle("celula", parent=padrao["Normal"], fontName="Corpo",
                                 fontSize=8.8, leading=12.4, textColor=TINTA),
        "celula_cab": ParagraphStyle("celula_cab", parent=padrao["Normal"], fontName="Corpo-Bold",
                                     fontSize=8.8, leading=12.4, textColor=colors.white),
    }


S = {}


def inline(texto: str) -> str:
    """Converte marcacao inline em tags do reportlab, escapando o resto."""
    fichas: list[str] = []

    def guardar(marcado: str) -> str:
        fichas.append(marcado)
        return f"\x00{len(fichas) - 1}\x00"

    texto = re.sub(
        r"`([^`]+)`",
        lambda m: guardar(f'<font face="Mono" size="8.6">{html.escape(m.group(1))}</font>'),
        texto,
    )
    # Links viram apenas o rotulo: o destino ja aparece por extenso no texto.
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)
    texto = html.escape(texto)
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", texto)
    return re.sub(r"\x00(\d+)\x00", lambda m: fichas[int(m.group(1))], texto)


def tabela(linhas: list[str]) -> Table:
    """
    Monta a tabela com colunas proporcionais ao conteudo de cada uma.

    Larguras fixas por numero de colunas nao servem: a regra que da a largura
    certa para uma coluna de indice ("#") esmaga uma coluna cujo cabecalho e
    "Parametros treinados", que entao quebra letra a letra.
    """
    linhas = [ln for ln in linhas if not re.match(r"^\|[\s|:-]+\|$", ln)]
    dados, cruas = [], []
    for i, linha in enumerate(linhas):
        celulas = [c.strip() for c in linha.strip().strip("|").split("|")]
        estilo = S["celula_cab"] if i == 0 else S["celula"]
        dados.append([Paragraph(inline(c), estilo) for c in celulas])
        cruas.append(celulas)

    n = max(len(linha) for linha in cruas)
    pesos = []
    for coluna in range(n):
        maior = max(
            (len(re.sub(r"[*`]", "", linha[coluna])) for linha in cruas if coluna < len(linha)),
            default=1,
        )
        # O teto impede que uma celula longa engula as demais; o piso garante
        # espaco para uma palavra inteira em cada coluna.
        pesos.append(min(max(maior, 6), 46))

    total = sum(pesos)
    colunas = [LARGURA_UTIL * peso / total for peso in pesos]

    t = Table(dados, colWidths=colunas, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, FUNDO]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def bloco_codigo(linhas: list[str]) -> Table:
    """
    Encaixa o bloco na largura util, encolhendo a fonte quando preciso.

    Sem isto uma URL longa - e o documento tem varias - transborda a caixa e sai
    cortada na margem, justamente onde o leitor precisa copiar o endereco
    inteiro.
    """
    texto = "\n".join(linhas)
    disponivel = LARGURA_UTIL - 18
    mais_longa = max((len(linha) for linha in linhas), default=1)

    corpo = S["codigo"]
    # Courier ocupa 0.6 em por caractere; abaixo de 5.5 pt deixa de ser legivel.
    cabe = disponivel / (mais_longa * 0.6)
    if cabe < corpo.fontSize:
        tamanho = max(5.5, round(cabe, 1))
        corpo = ParagraphStyle(
            "codigo_ajustado", parent=corpo, fontSize=tamanho, leading=tamanho * 1.34
        )

    t = Table([[Preformatted(texto, corpo)]], colWidths=[LARGURA_UTIL], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FUNDO),
        ("BOX", (0, 0), (-1, -1), 0.5, LINHA),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def resolver_inclusoes(texto: str, base: Path) -> str:
    """Substitui `{{incluir: arquivo.md}}` pelo conteudo do arquivo."""

    def trocar(achado: re.Match[str]) -> str:
        caminho = base / achado.group(1).strip()
        if not caminho.exists():
            raise SystemExit(f"guia: arquivo incluido nao encontrado: {caminho}")
        corpo = caminho.read_text(encoding="utf-8")
        # O documento incluido tem titulo e linha de identificacao proprios: no
        # guia eles seriam repeticao, e o titulo da secao vem do arquivo que
        # faz a inclusao.
        corpo = re.sub(r"\A#\s+.*?\n", "", corpo, count=1)
        corpo = re.sub(r"\A\s*\*\*[^\n]*\*\*\s*\n", "", corpo, count=1)
        # O conversor nao desenha caixa de selecao.
        return corpo.replace("- [ ] ", "- ")

    return re.sub(r"\{\{\s*incluir:\s*([^}]+)\}\}", trocar, texto)


def construir(markdown: str) -> list:
    linhas = markdown.splitlines()
    fluxo: list = []
    buffer: list[str] = []
    item: list[str] = []
    i = indice_h1 = indice_h2 = 0
    na_capa = True

    def fechar_item() -> None:
        """
        Um item de lista pode ocupar varias linhas.

        No Markdown a continuacao vem indentada, sem novo hifen. Emitir o item
        assim que o hifen aparece deixaria a continuacao virar um paragrafo
        solto, fora da marca da lista.
        """
        nonlocal item
        if not item:
            return
        texto = " ".join(linha.strip() for linha in item)
        fluxo.append(Paragraph(inline(texto), S["item"], bulletText="•"))
        item = []

    def descarregar() -> None:
        """Linhas consecutivas formam UM paragrafo - a quebra e a linha vazia."""
        nonlocal buffer
        fechar_item()
        if not buffer:
            return
        texto = " ".join(linha.strip() for linha in buffer)
        fluxo.append(Paragraph(inline(texto), S["capa_nota"] if na_capa else S["corpo"]))
        buffer = []

    while i < len(linhas):
        linha = linhas[i]

        if linha.startswith("```"):
            descarregar()
            corpo: list[str] = []
            i += 1
            while i < len(linhas) and not linhas[i].startswith("```"):
                corpo.append(linhas[i])
                i += 1
            fluxo += [Spacer(1, 4), bloco_codigo(corpo), Spacer(1, 10)]
            i += 1
            continue

        if linha.startswith("|"):
            descarregar()
            corpo = []
            while i < len(linhas) and linhas[i].startswith("|"):
                corpo.append(linhas[i])
                i += 1
            fluxo += [Spacer(1, 4), tabela(corpo), Spacer(1, 12)]
            continue

        if linha.startswith("# "):
            descarregar()
            indice_h1 += 1
            if indice_h1 == 1:
                fluxo += [Spacer(1, 4.5 * cm), Paragraph(inline(linha[2:]), S["titulo"])]
            else:
                na_capa = False
                indice_h2 = 0
                # A numeracao e automatica para que inserir uma secao no meio do
                # documento nao exija renumerar tudo a mao.
                fluxo += [
                    PageBreak(),
                    Paragraph(inline(f"{indice_h1 - 1}. {linha[2:]}"), S["h1"]),
                ]
        elif linha.startswith("## "):
            descarregar()
            if na_capa:
                fluxo.append(Paragraph(inline(linha[3:]), S["subtitulo"]))
            else:
                indice_h2 += 1
                fluxo += [
                    CondPageBreak(3 * cm),
                    Paragraph(inline(f"{indice_h1 - 1}.{indice_h2} {linha[3:]}"), S["h2"]),
                ]
        elif linha.startswith("### "):
            descarregar()
            fluxo += [CondPageBreak(2.5 * cm), Paragraph(inline(linha[4:]), S["h3"])]
        elif linha.startswith("> "):
            descarregar()
            fluxo.append(Paragraph(inline(linha[2:]), S["citacao"]))
        elif linha.startswith("- "):
            descarregar()
            item = [linha[2:]]
        elif item and linha[:1].isspace() and linha.strip():
            # Linha indentada logo apos um item: e continuacao dele.
            item.append(linha)
        elif linha.strip() == "---":
            descarregar()
            na_capa = False if na_capa and indice_h1 else na_capa
            fluxo.append(Spacer(1, 6))
        elif linha.strip():
            buffer.append(linha)
        else:
            descarregar()
        i += 1

    descarregar()
    return fluxo


def rodape(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Corpo", 8)
    canvas.setFillColor(SUAVE)
    canvas.drawString(2.3 * cm, 1.5 * cm, "MedGraph — Guia de entendimento do projeto")
    canvas.drawRightString(A4[0] - 2.3 * cm, 1.5 * cm, str(doc.page))
    canvas.setStrokeColor(LINHA)
    canvas.setLineWidth(0.4)
    canvas.line(2.3 * cm, 2.0 * cm, A4[0] - 2.3 * cm, 2.0 * cm)
    canvas.restoreState()


def main() -> int:
    if not registrar_fontes():
        print(
            "As fontes do sistema nao foram encontradas em\n"
            f"  {FONTES_MAC}\n"
            "O gerador depende delas para acentuacao e travessao. Em outro "
            "sistema, ajuste FONTES_MAC no topo deste script.",
            file=sys.stderr,
        )
        return 1

    S.update(montar_estilos())

    if not FONTE.exists():
        print(f"fonte do guia nao encontrada: {FONTE}", file=sys.stderr)
        return 1

    markdown = resolver_inclusoes(FONTE.read_text(encoding="utf-8"), FONTE.parent)

    doc = BaseDocTemplate(
        str(SAIDA), pagesize=A4,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=2.2 * cm, bottomMargin=2.5 * cm,
        title="MedGraph - Guia de entendimento do projeto",
        author="Grupo 8IADT",
    )
    quadro = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="corpo")
    doc.addPageTemplates([PageTemplate(id="padrao", frames=[quadro], onPage=rodape)])
    doc.build(construir(markdown))

    tamanho = SAIDA.stat().st_size / 1024
    print(f"Guia gerado em {SAIDA.relative_to(RAIZ)}  ({tamanho:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
