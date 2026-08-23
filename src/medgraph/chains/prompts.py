"""
[REQ-2][REQ-3a][REQ-3c] Prompts do MedGraph.

O QUE FAZ:
    Concentra todos os textos de instrucao enviados ao modelo - o prompt de
    sistema, o formato de apresentacao do contexto recuperado e os moldes de
    cada tarefa.

POR QUE ISSO E CRITICO, E NAO APENAS ORGANIZACAO:
    O prompt de sistema definido aqui e usado em DOIS momentos que precisam
    coincidir exatamente:

      1. na construcao do dataset de fine-tuning (Etapa 1/2);
      2. em toda chamada ao modelo em producao (Etapas 5 a 7).

    Se os dois divergirem, o modelo e treinado sob um contrato e consultado
    sob outro. O sintoma tipico e cruel: o modelo funciona bem na avaliacao
    offline e se comporta de forma erratica no fluxo real - com destaque para
    o abandono do formato de citacao, que e justamente o que sustenta o
    requisito de explainability.

    Manter um unico arquivo importado pelos dois lados torna a divergencia
    impossivel por construcao.

CONVENCAO DE CITACAO:
    Toda afirmacao clinica precisa apontar sua origem, usando um destes
    marcadores:

      [E#]  evidencia cientifica (abstract do PubMedQA)
      [P#]  protocolo interno do Hospital Vida Plena
      [C#]  dado clinico do proprio paciente (prontuario)

    A escolha de marcadores curtos e deliberada: sao faceis de o modelo
    reproduzir, faceis de validar com expressao regular no guardrail de
    saida, e faceis de resolver de volta para a fonte completa no painel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# =============================================================================
# PROMPT DE SISTEMA
# =============================================================================
# Estrutura em quatro blocos: quem e, o que faz, o que NAO faz, como responde.
# A ordem importa - os limites vem antes do formato porque, quando o modelo
# precisa escolher entre obedecer ao formato e obedecer a um limite de
# seguranca, queremos que o limite prevaleca.
# =============================================================================
SISTEMA = """Você é o MedGraph, assistente de apoio à decisão clínica do Hospital Vida Plena.

QUEM CONSULTA VOCÊ
Você atende exclusivamente o corpo clínico — médicos e residentes. Nunca atende pacientes.
Quem lê sua resposta tem formação médica e exercerá julgamento clínico próprio.

O QUE VOCÊ FAZ
- Responde dúvidas clínicas com base na evidência científica e nos protocolos internos fornecidos.
- Contextualiza a resposta com os dados do paciente quando eles forem fornecidos.
- Aponta conflitos entre a conduta discutida e alergias, medicações em uso ou exames do paciente.

LIMITES DE ATUAÇÃO — inegociáveis
1. Você NÃO prescreve. Pode apresentar o que dizem os protocolos e a evidência sobre as opções
   terapêuticas, mas a prescrição é ato privativo do médico responsável.
2. Você NÃO responde sem fonte. Se o contexto fornecido não sustentar uma afirmação, diga
   explicitamente que não há informação suficiente.
3. Você NÃO inventa dados de paciente, valores de exame, doses ou referências.
4. Você NÃO trata assuntos fora do escopo clínico.

COMO VOCÊ RESPONDE
- Em português do Brasil, objetivo e direto, sem rodeios nem repetição da pergunta.
- Toda afirmação clínica cita a fonte com um marcador: [E#] para evidência científica,
  [P#] para protocolo interno, [C#] para dado do prontuário do paciente.
- Ao final, uma linha "Fontes:" listando os marcadores usados.
- Quando mencionar dose, via ou posologia, marque explicitamente que depende de validação médica."""


# =============================================================================
# TAREFA: RACIOCINIO SOBRE EVIDENCIA (PubMedQA)
# =============================================================================
# O formato de saida e fixo: uma linha "Decisão:" com um dos tres rotulos,
# depois a justificativa. Fixar a primeira linha torna a extracao do rotulo
# na avaliacao triviais e deterministica - nao precisamos de um segundo
# modelo para interpretar a resposta do primeiro.
INSTRUCAO_DECISAO = """Com base exclusivamente na evidência fornecida, responda à pergunta de pesquisa.

Formato obrigatório da resposta:
Decisão: yes | no | maybe
<justificativa em 1 a 3 frases, citando a evidência>
Fontes: [E1]

Use "maybe" quando a evidência for inconclusiva, parcial ou contraditória — não force uma
resposta definitiva onde o estudo não a sustenta."""


def montar_contexto(fontes: Iterable[Mapping[str, str]]) -> str:
    """
    Formata as fontes recuperadas no bloco que o modelo le.

    Cada fonte precisa ter as chaves:
        marcador  ex.: "E1", "P2", "C1"
        titulo    identificacao curta e legivel da origem
        texto     o trecho propriamente dito

    O marcador vem primeiro na linha, de proposito: e o token que o modelo
    precisa copiar para a resposta, e posicoes iniciais sao mais faceis de
    o mecanismo de atencao recuperar.
    """
    blocos: list[str] = []
    for fonte in fontes:
        blocos.append(
            f"[{fonte['marcador']}] {fonte['titulo']}\n{fonte['texto'].strip()}"
        )
    return "\n\n".join(blocos)


def usuario_decisao(pergunta: str, contexto: str, marcador: str = "E1") -> str:
    """Mensagem do usuario na tarefa de decisao sobre evidencia cientifica."""
    return (
        f"{INSTRUCAO_DECISAO}\n\n"
        f"Pergunta: {pergunta.strip()}\n\n"
        f"Evidência disponível:\n[{marcador}] Abstract PubMed\n{contexto.strip()}"
    )


def assistente_decisao(decisao: str, justificativa: str, marcador: str = "E1") -> str:
    """Resposta de referencia na tarefa de decisao. Usada no treino."""
    return (
        f"Decisão: {decisao.strip().lower()}\n"
        f"{justificativa.strip()}\n"
        f"Fontes: [{marcador}]"
    )


# =============================================================================
# TAREFA: DUVIDA CLINICA SOBRE PROTOCOLO INTERNO
# =============================================================================
INSTRUCAO_PROTOCOLO = """Responda à dúvida do médico com base no protocolo interno fornecido.

Cite o protocolo com o marcador correspondente. Se a resposta envolver dose, via ou posologia,
inclua a marcação de que depende de validação do médico responsável."""


def usuario_protocolo(pergunta: str, contexto: str) -> str:
    return f"{INSTRUCAO_PROTOCOLO}\n\nPergunta: {pergunta.strip()}\n\nContexto:\n{contexto.strip()}"


# =============================================================================
# TAREFA: GERACAO DE DOCUMENTO INSTITUCIONAL
# =============================================================================
INSTRUCAO_DOCUMENTO = """Apresente o modelo institucional do documento solicitado, com todos os
campos obrigatórios. Não preencha dados de paciente que não tenham sido fornecidos.

Se o documento for prescrição ou receita, deixe explícito que o texto é um rascunho e que só
tem validade após assinatura do médico responsável."""


def usuario_documento(pedido: str, contexto: str) -> str:
    return f"{INSTRUCAO_DOCUMENTO}\n\nPedido: {pedido.strip()}\n\nModelo institucional:\n{contexto.strip()}"


# =============================================================================
# RECUSAS
# =============================================================================
# Treinar o modelo a recusar bem e tao importante quanto treina-lo a
# responder bem. Um modelo que so viu exemplos de resposta tenta responder
# tudo, inclusive o que nao deveria.
INSTRUCAO_RECUSA = (
    "Responda ao pedido do usuário respeitando seus limites de atuação."
)
