"""
[REQ-1a] Curadoria e divisao do PubMedQA.

O QUE FAZ:
    Transforma os registros brutos em um conjunto de dados confiavel:
    remove duplicatas, descarta registros incompletos ou degenerados,
    aplica anonimizacao, e divide o material anotado por especialistas em
    treino, validacao e teste - de forma estratificada e reproduzivel.

POR QUE CURADORIA IMPORTA MAIS DO QUE PARECE:
    O enunciado pede "preprocessing, anonimizacao e curadoria" como se fossem
    tres tarefas equivalentes. Na pratica, a curadoria e a que mais afeta o
    resultado final. Um dataset com duplicatas infla artificialmente a
    metrica de teste (o modelo ja viu a resposta no treino); um dataset com
    contexto vazio ensina o modelo a chutar; um dataset desbalanceado faz o
    modelo aprender a responder sempre a classe majoritaria e ainda assim
    parecer razoavel na accuracy.

    Cada filtro aqui responde a um desses riscos, e todos deixam rastro no
    relatorio de curadoria - quantos registros sairam e por qual motivo.

A DIVISAO E O PONTO MAIS DELICADO:
    O subconjunto anotado por especialistas (pqa_labeled, 1.000 registros) e
    dividido em 450 treino / 50 validacao / 500 teste. Esses 500 de teste
    NUNCA entram no fine-tuning, em nenhuma forma. E o unico jeito de a
    avaliacao da Etapa 4 significar alguma coisa: medir um modelo em dados
    que ele ja viu mede memorizacao, nao capacidade.

    A divisao e ESTRATIFICADA por rotulo (yes/no/maybe) para que as tres
    classes aparecam na mesma proporcao nos tres conjuntos, e usa semente
    fixa para ser reproduzivel por quem clonar o repositorio.

Uso:
    python -m medgraph.dados.curadoria
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, obter_settings
from medgraph.auditoria import TipoEvento, etapa, registrar
from medgraph.dados import baixar_pubmedqa
from medgraph.dados.anonimizador import Anonimizador, Politica
from medgraph.logging_config import obter_logger

log = obter_logger(__name__)

ROTULOS_VALIDOS = frozenset({"yes", "no", "maybe"})

# Caracteres de controle e separadores exoticos que aparecem nos abstracts
# por causa da conversao de PDF para texto. Sao inofensivos para um leitor
# humano e destrutivos para um arquivo JSON Lines: o json.dumps NAO os
# escapa, mas varias funcoes de leitura os tratam como quebra de linha -
# partindo um registro em dois. Removemos na origem.
CARACTERES_QUEBRA_INVISIVEIS = str.maketrans(
    {c: " " for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"}
)


def higienizar(texto: str) -> str:
    """Normaliza espacos e remove separadores de linha invisiveis."""
    if not texto:
        return texto
    limpo = texto.translate(CARACTERES_QUEBRA_INVISIVEIS)
    # Colapsa sequencias de espaco/tabulacao sem destruir a quebra de
    # paragrafo, que carrega a estrutura do abstract.
    limpo = re.sub(r"[ \t]+", " ", limpo)
    limpo = re.sub(r"\n{3,}", "\n\n", limpo)
    return limpo.strip()

# Limites dos filtros de qualidade. Nao sao arbitrarios: foram escolhidos
# depois de inspecionar a distribuicao real do dataset, e cada um esta
# justificado no comentario do filtro correspondente.
MIN_CARACTERES_CONTEXTO = 200
MIN_CARACTERES_PERGUNTA = 15
MIN_CARACTERES_RESPOSTA = 40
MAX_CARACTERES_CONTEXTO = 12_000

# Proporcao da divisao do conjunto anotado por especialistas.
PROPORCAO_TESTE = 0.50
PROPORCAO_VALIDACAO = 0.05

# Quantas vezes a classe majoritaria pode, no maximo, superar a minoritaria
# no conjunto artificial. Ver balancear_por_rotulo() para a justificativa.
RAZAO_MAXIMA_ENTRE_CLASSES = 3.0


@dataclass
class RelatorioCuradoria:
    """
    Contabilidade do que entrou, do que saiu e por que.

    Este objeto vira uma secao do relatorio tecnico. Sem ele, a curadoria
    seria uma caixa-preta: o leitor veria o numero final sem poder julgar se
    os criterios foram razoaveis.
    """

    subconjunto: str
    total_entrada: int = 0
    descartados: Counter[str] = field(default_factory=Counter)
    duplicatas: int = 0
    truncados_para_amostra: int = 0
    """Registros que PASSARAM nos filtros mas sobraram do tamanho de amostra pedido."""
    removidos_por_balanceamento: int = 0
    """Registros da classe majoritaria removidos para conter o desbalanceamento."""
    identificadores_removidos: dict[str, int] = field(default_factory=dict)
    distribuicao_final: Counter[str] = field(default_factory=Counter)

    @property
    def total_saida(self) -> int:
        return sum(self.distribuicao_final.values())

    @property
    def total_reprovado_por_filtro(self) -> int:
        return sum(self.descartados.values()) + self.duplicatas

    @property
    def taxa_aprovacao_nos_filtros(self) -> float:
        """
        Proporcao dos registros que PASSARAM nos criterios de qualidade.

        Deliberadamente separada do tamanho final: cortar a amostra para
        8.000 exemplos nao e reprovacao de qualidade, e escolha de escopo.
        """
        if not self.total_entrada:
            return 0.0
        return (self.total_entrada - self.total_reprovado_por_filtro) / self.total_entrada

    def para_dict(self) -> dict[str, Any]:
        return {
            "subconjunto": self.subconjunto,
            "total_entrada": self.total_entrada,
            "total_saida": self.total_saida,
            "reprovados_por_filtro": self.total_reprovado_por_filtro,
            "taxa_aprovacao_nos_filtros": round(self.taxa_aprovacao_nos_filtros, 4),
            "truncados_para_amostra": self.truncados_para_amostra,
            "removidos_por_balanceamento": self.removidos_por_balanceamento,
            "duplicatas_removidas": self.duplicatas,
            "descartados_por_motivo": dict(self.descartados),
            "identificadores_anonimizados": self.identificadores_removidos,
            "distribuicao_de_rotulos": dict(self.distribuicao_final),
        }

    def resumo_texto(self) -> str:
        linhas = [
            f"Curadoria de {self.subconjunto}",
            f"  entrada ............ {self.total_entrada:>7,}",
            f"  duplicatas ......... {self.duplicatas:>7,}",
        ]
        for motivo, qtd in sorted(self.descartados.items(), key=lambda x: -x[1]):
            linhas.append(f"  {motivo:.<19} {qtd:>7,}")
        linhas.append(
            f"  aprovados nos filtros {self.total_entrada - self.total_reprovado_por_filtro:>5,}"
            f"  ({self.taxa_aprovacao_nos_filtros:.1%})"
        )
        if self.removidos_por_balanceamento:
            linhas.append(f"  balanceamento ...... {self.removidos_por_balanceamento:>7,}")
        if self.truncados_para_amostra:
            linhas.append(f"  corte de amostra ... {self.truncados_para_amostra:>7,}")
        linhas.append(f"  SAIDA .............. {self.total_saida:>7,}")
        linhas.append(f"  rotulos ............ {dict(self.distribuicao_final)}")
        if self.identificadores_removidos:
            linhas.append(f"  PII removida ....... {self.identificadores_removidos}")
        return "\n".join(linhas)


def _chave_deduplicacao(registro: dict[str, Any]) -> str:
    """
    Assinatura de um registro, para detectar repeticao.

    Usamos pergunta + contexto, e nao o pubid. Dois artigos diferentes podem
    gerar a MESMA pergunta com o MESMO abstract (acontece com errata e com
    republicacao), e nesse caso o pubid difere mas o conteudo e identico -
    exatamente o tipo de duplicata que contamina a avaliacao.
    """
    bruto = f"{registro['question'].lower().strip()}||{registro['contexto_texto'][:500].lower()}"
    return hashlib.sha256(bruto.encode()).hexdigest()


def curar(
    registros: list[dict[str, Any]],
    *,
    subconjunto: str,
    anonimizar: bool = True,
    cfg: Settings | None = None,
) -> tuple[list[dict[str, Any]], RelatorioCuradoria]:
    """
    Aplica os filtros de qualidade e a anonimizacao.

    Returns:
        Os registros aprovados e o relatorio do que aconteceu.
    """
    cfg = cfg or obter_settings()
    relatorio = RelatorioCuradoria(subconjunto=subconjunto, total_entrada=len(registros))

    # O PubMedQA e um corpus cientifico em ingles: nao esperamos encontrar
    # PII de paciente ali. Passamos o anonimizador mesmo assim por duas
    # razoes: e-mails de autores correspondentes aparecem em alguns abstracts,
    # e - mais importante - o pipeline precisa ser o MESMO que sera aplicado
    # a dados hospitalares reais. Um pipeline que so anonimiza quando alguem
    # lembra de ligar nao e um pipeline de anonimizacao.
    anonimizador = Anonimizador(politica=Politica.MASCARAR) if anonimizar else None

    vistos: set[str] = set()
    aprovados: list[dict[str, Any]] = []

    for registro in registros:
        pergunta = registro.get("question", "")
        contexto = registro.get("contexto_texto", "")
        resposta = registro.get("long_answer", "")
        rotulo = registro.get("final_decision", "")

        # --- Filtro 1: rotulo valido -------------------------------------
        # Sem rotulo confiavel nao ha supervisao. Registros com rotulo vazio
        # ou fora do vocabulario nao ensinam nada e envenenam a metrica.
        if rotulo not in ROTULOS_VALIDOS:
            relatorio.descartados["rotulo_invalido"] += 1
            continue

        # --- Filtro 2: contexto utilizavel -------------------------------
        # Abaixo de 200 caracteres nao ha abstract, ha um fragmento. O modelo
        # aprenderia a responder sem evidencia - o oposto do comportamento
        # que este projeto quer induzir.
        if len(contexto) < MIN_CARACTERES_CONTEXTO:
            relatorio.descartados["contexto_curto_demais"] += 1
            continue

        # Acima de 12.000 caracteres o exemplo nao cabe na janela de contexto
        # do treino (1.024 tokens) sem truncamento agressivo, que cortaria
        # justamente a conclusao do abstract.
        if len(contexto) > MAX_CARACTERES_CONTEXTO:
            relatorio.descartados["contexto_longo_demais"] += 1
            continue

        # --- Filtro 3: pergunta e resposta com substancia ----------------
        if len(pergunta) < MIN_CARACTERES_PERGUNTA:
            relatorio.descartados["pergunta_curta_demais"] += 1
            continue
        if len(resposta) < MIN_CARACTERES_RESPOSTA:
            relatorio.descartados["resposta_curta_demais"] += 1
            continue

        # --- Filtro 4: duplicatas ----------------------------------------
        chave = _chave_deduplicacao(registro)
        if chave in vistos:
            relatorio.duplicatas += 1
            continue
        vistos.add(chave)

        # --- Anonimizacao -------------------------------------------------
        if anonimizador is not None:
            pergunta = anonimizador.redigir(pergunta)
            contexto = anonimizador.redigir(contexto)
            resposta = anonimizador.redigir(resposta)

        aprovados.append(
            {
                "pubid": registro.get("pubid"),
                "pergunta": higienizar(pergunta),
                "contexto": higienizar(contexto),
                "resposta_longa": higienizar(resposta),
                "decisao": rotulo,
                "mesh": registro.get("mesh", [])[:8],
            }
        )
        relatorio.distribuicao_final[rotulo] += 1

    if anonimizador is not None:
        relatorio.identificadores_removidos = anonimizador.estatisticas()

    return aprovados, relatorio


def dividir_estratificado(
    registros: list[dict[str, Any]],
    *,
    semente: int,
    proporcao_teste: float = PROPORCAO_TESTE,
    proporcao_validacao: float = PROPORCAO_VALIDACAO,
) -> dict[str, list[dict[str, Any]]]:
    """
    Divide em treino/validacao/teste preservando a proporcao de rotulos.

    POR QUE ESTRATIFICADO:
        O PubMedQA e desbalanceado - "maybe" e a classe minoritaria, com
        cerca de 11% dos casos. Uma divisao aleatoria simples poderia deixar
        pouquissimos "maybe" no teste, e a metrica dessa classe viraria ruido
        estatistico. Estratificar garante que as tres classes apareçam na
        mesma proporcao nos tres conjuntos.

    POR QUE SEMENTE FIXA:
        Para que qualquer pessoa que clone o repositorio obtenha exatamente
        a mesma divisao e possa reproduzir os numeros do relatorio tecnico.
    """
    por_rotulo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in registros:
        por_rotulo[r["decisao"]].append(r)

    rng = random.Random(semente)
    divisao: dict[str, list[dict[str, Any]]] = {"treino": [], "validacao": [], "teste": []}

    for _rotulo, itens in sorted(por_rotulo.items()):
        embaralhados = itens[:]
        rng.shuffle(embaralhados)

        n = len(embaralhados)
        n_teste = int(round(n * proporcao_teste))
        n_validacao = max(1, int(round(n * proporcao_validacao))) if n > 20 else 0

        divisao["teste"].extend(embaralhados[:n_teste])
        divisao["validacao"].extend(embaralhados[n_teste : n_teste + n_validacao])
        divisao["treino"].extend(embaralhados[n_teste + n_validacao :])

    # Embaralha novamente dentro de cada conjunto: sem isso os registros
    # ficariam agrupados por rotulo, o que atrapalha o treino em lote.
    for conjunto in divisao.values():
        rng.shuffle(conjunto)

    return divisao


def balancear_por_rotulo(
    registros: list[dict[str, Any]],
    *,
    razao_maxima: float,
    semente: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Limita o quanto a classe majoritaria pode dominar o conjunto.

    POR QUE ISSO E NECESSARIO AQUI:
        O subconjunto pqa_artificial do PubMedQA tem os rotulos gerados
        automaticamente, e o resultado e fortemente enviesado: cerca de 93%
        das respostas sao "yes" e nao existe nenhum "maybe". Um modelo
        treinado nessa proporcao aprende o atalho errado - responder "yes"
        sempre - e ainda assim exibe uma accuracy aparentemente boa.

        Recortamos a classe majoritaria ate que ela seja, no maximo,
        `razao_maxima` vezes maior que a minoritaria. Preferimos subamostrar
        a majoritaria em vez de replicar a minoritaria: duplicar exemplos
        raros faz o modelo memoriza-los, o que e outra forma de trapaca.

    NOTA IMPORTANTE:
        O "maybe" ausente aqui vem do conjunto anotado por especialistas
        (pqa_labeled), que tem as tres classes. Essa complementaridade e a
        razao de usarmos os dois subconjuntos, e nao apenas o maior.

    Returns:
        Os registros balanceados e quantos foram removidos.
    """
    por_rotulo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in registros:
        por_rotulo[r["decisao"]].append(r)

    if len(por_rotulo) < 2:
        return registros, 0

    menor = min(len(v) for v in por_rotulo.values())
    teto = int(menor * razao_maxima)

    rng = random.Random(semente)
    balanceados: list[dict[str, Any]] = []
    removidos = 0
    for _, itens in sorted(por_rotulo.items()):
        if len(itens) > teto:
            removidos += len(itens) - teto
            itens = rng.sample(itens, teto)
        balanceados.extend(itens)

    rng.shuffle(balanceados)
    return balanceados, removidos


def _gravar_jsonl(registros: list[dict[str, Any]], caminho) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for r in registros:
            arquivo.write(json.dumps(r, ensure_ascii=False) + "\n")


def executar(cfg: Settings | None = None) -> dict[str, Any]:
    """
    Roda a curadoria completa e grava os conjuntos em data/processed/.

    Arquivos produzidos:
        pubmedqa_treino.jsonl      conjunto anotado, parte de treino
        pubmedqa_validacao.jsonl   conjunto anotado, parte de validacao
        pubmedqa_teste.jsonl       conjunto anotado, INTOCADO ate a Etapa 4
        pubmedqa_artificial.jsonl  amostra do subconjunto automatico
        relatorio_curadoria.json   contabilidade completa
    """
    cfg = cfg or obter_settings()
    relatorios: list[RelatorioCuradoria] = []

    # ---------------------------------------------------------------------
    # Conjunto anotado por especialistas
    # ---------------------------------------------------------------------
    with etapa("curadoria:pqa_labeled"):
        brutos = baixar_pubmedqa.carregar("pqa_labeled", cfg)
        curados, relatorio = curar(brutos, subconjunto="pqa_labeled", cfg=cfg)
        relatorios.append(relatorio)
        log.info("\n%s", relatorio.resumo_texto())

        divisao = dividir_estratificado(curados, semente=cfg.semente_aleatoria)
        for nome, itens in divisao.items():
            _gravar_jsonl(itens, cfg.dir_dados_processados / f"pubmedqa_{nome}.jsonl")
            log.info(
                "  %s: %d registros | %s",
                nome,
                len(itens),
                dict(Counter(i["decisao"] for i in itens)),
            )

    # ---------------------------------------------------------------------
    # Conjunto artificial (volume de treino)
    # ---------------------------------------------------------------------
    with etapa("curadoria:pqa_artificial"):
        # AMOSTRAGEM POR RESERVATORIO ESTRATIFICADA
        #
        # Dois problemas resolvidos de uma vez:
        #
        #   TAMANHO - o arquivo tem 211 mil registros e ~735 MB. Percorremos
        #   linha a linha, mantendo reservatorios em memoria; o consumo
        #   depende do tamanho da amostra, nao do arquivo.
        #
        #   DESBALANCEAMENTO - os rotulos automaticos sao 93% "yes". Uma
        #   amostragem uniforme reproduziria esse vies e, depois do
        #   balanceamento, sobraria uma fracao do volume pretendido. Manter um
        #   reservatorio SEPARADO POR ROTULO, cada um ja dimensionado na
        #   proporcao desejada, sorteia diretamente o conjunto que queremos.
        #
        # Cada registro de um dado rotulo continua tendo a mesma probabilidade
        # de entrar no reservatorio daquele rotulo, entao a amostra permanece
        # aleatoria dentro de cada classe.
        rng = random.Random(cfg.semente_aleatoria)
        alvo = cfg.pubmedqa_artificial_amostra

        # Reparticao do alvo entre as classes, respeitando a razao maxima.
        # Com razao 3.0: 3 partes de "yes" para 1 de "no" -> 75% / 25%.
        partes = RAZAO_MAXIMA_ENTRE_CLASSES + 1.0
        cota = {
            "yes": int(alvo * RAZAO_MAXIMA_ENTRE_CLASSES / partes * 1.15),
            "no": int(alvo * 1.0 / partes * 1.15),
        }  # 15% de folga para o que os filtros de qualidade vao reprovar

        reservatorios: dict[str, list[dict[str, Any]]] = {r: [] for r in cota}
        contador_fluxo: Counter[str] = Counter()

        for registro in baixar_pubmedqa.iterar("pqa_artificial", cfg):
            rotulo = (registro.get("final_decision") or "").strip().lower()
            if rotulo not in cota:
                continue
            contador_fluxo[rotulo] += 1
            reservatorio = reservatorios[rotulo]
            limite = cota[rotulo]

            if len(reservatorio) < limite:
                reservatorio.append(registro)
            else:
                # Substituicao com probabilidade limite/n - o algoritmo
                # classico de Vitter, aplicado por classe.
                posicao = rng.randint(0, contador_fluxo[rotulo] - 1)
                if posicao < limite:
                    reservatorio[posicao] = registro

        amostra = [r for reservatorio in reservatorios.values() for r in reservatorio]
        rng.shuffle(amostra)
        log.info(
            "Reservatorio estratificado: %s sorteados de %s disponiveis no fluxo",
            {r: len(v) for r, v in reservatorios.items()},
            dict(contador_fluxo),
        )

        curados_art, relatorio_art = curar(amostra, subconjunto="pqa_artificial", cfg=cfg)

        curados_art, removidos = balancear_por_rotulo(
            curados_art,
            razao_maxima=RAZAO_MAXIMA_ENTRE_CLASSES,
            semente=cfg.semente_aleatoria,
        )
        relatorio_art.removidos_por_balanceamento = removidos

        if len(curados_art) > alvo:
            relatorio_art.truncados_para_amostra = len(curados_art) - alvo
            curados_art = curados_art[:alvo]

        relatorio_art.distribuicao_final = Counter(r["decisao"] for r in curados_art)
        relatorios.append(relatorio_art)
        log.info("\n%s", relatorio_art.resumo_texto())

        _gravar_jsonl(curados_art, cfg.dir_dados_processados / "pubmedqa_artificial.jsonl")

    # ---------------------------------------------------------------------
    # Relatorio
    # ---------------------------------------------------------------------
    resumo = {
        "semente": cfg.semente_aleatoria,
        "relatorios": [r.para_dict() for r in relatorios],
        "divisao_anotado": {nome: len(itens) for nome, itens in divisao.items()},
        "artificial": len(curados_art),
    }
    destino = cfg.dir_dados_processados / "relatorio_curadoria.json"
    destino.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")

    registrar(
        TipoEvento.FIM_ETAPA,
        "Curadoria concluida",
        etapa="curadoria",
        conclusao=True,
        **{f"{k}": v for k, v in resumo["divisao_anotado"].items()},
        artificial=len(curados_art),
    )
    log.info("Relatorio de curadoria gravado em %s", destino)
    return resumo


if __name__ == "__main__":
    from medgraph import iniciar

    iniciar(banner="Curadoria do PubMedQA", subtitulo="filtros de qualidade, anonimizacao e divisao")
    executar()
