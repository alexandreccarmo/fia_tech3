"""
Dados do assistente: PubMedQA para o fine-tuning, protocolos para o RAG.

O enunciado pede tres tipos de material - protocolos do hospital, perguntas
frequentes de medicos e modelos de laudo, receita e procedimento. Os tres
estao aqui: o PubMedQA entra
como base de evidencia cientifica real, e o material do hospital e sintetico,
porque base hospitalar real nao e distribuivel (o proprio enunciado aceita
"dataset anonimizado ou exemplo de dados sinteticos").
"""

from __future__ import annotations

import random
import re

# =============================================================================
# PROTOCOLOS DO HOSPITAL VIDA PLENA  (sinteticos)
# =============================================================================
PROTOCOLOS = [
    {
        "id": "P1",
        "titulo": "Sepse - primeira hora",
        "texto": (
            "Na suspeita de sepse, colher lactato e hemoculturas antes do antibiotico. "
            "Iniciar antibiotico de amplo espectro em ate 1 hora. Reposicao volemica de "
            "30 mL/kg de cristaloide se hipotensao ou lactato acima de 4 mmol/L. "
            "Reavaliar perfusao apos a expansao."
        ),
    },
    {
        "id": "P2",
        "titulo": "Pneumonia adquirida na comunidade",
        "texto": (
            "Estratificar gravidade pelo CURB-65. Para internacao em enfermaria, "
            "betalactamico associado a macrolideo. Em alergia a penicilina, usar "
            "quinolona respiratoria. Reavaliar em 48 a 72 horas."
        ),
    },
    {
        "id": "P3",
        "titulo": "Ajuste de dose na disfuncao renal",
        "texto": (
            "Calcular a taxa de filtracao glomerular antes de prescrever antimicrobianos "
            "de excrecao renal. Com clearance abaixo de 30 mL/min, ajustar dose e "
            "intervalo. Evitar anti-inflamatorios nao esteroidais."
        ),
    },
    {
        "id": "P4",
        "titulo": "Alergia a betalactamicos",
        "texto": (
            "Penicilinas, cefalosporinas e carbapenemicos compartilham o anel "
            "betalactamico. Em historia de anafilaxia a penicilina, evitar a classe "
            "inteira. Alternativas: quinolonas, macrolideos, glicopeptideos."
        ),
    },
    {
        "id": "P5",
        "titulo": "Anticoagulacao e interacoes",
        "texto": (
            "Varfarina tem interacao relevante com amiodarona, sulfametoxazol e "
            "fluconazol, que elevam o INR. Reavaliar o INR em 3 a 5 dias apos "
            "introduzir qualquer um deles."
        ),
    },
]

# Perguntas frequentes do corpo medico, com a resposta no formato que queremos
# ensinar ao modelo.
FAQ = [
    (
        "Qual o tempo maximo para iniciar antibiotico na sepse?",
        "Decisao: yes\nAte 1 hora do reconhecimento, apos colher hemoculturas e lactato. [P1]",
    ),
    (
        "Posso usar cefalosporina em paciente com anafilaxia a penicilina?",
        "Decisao: no\nCefalosporinas compartilham o anel betalactamico com as penicilinas; "
        "em anafilaxia, evitar a classe inteira. [P4]",
    ),
    (
        "Preciso ajustar antimicrobiano quando o clearance esta em 25 mL/min?",
        "Decisao: yes\nAbaixo de 30 mL/min, ajustar dose e intervalo dos antimicrobianos "
        "de excrecao renal. [P3]",
    ),
    (
        "Amiodarona interfere na varfarina?",
        "Decisao: yes\nEleva o INR; reavaliar em 3 a 5 dias apos a introducao. [P5]",
    ),
    (
        "CURB-65 de 1 indica internacao em UTI?",
        "Decisao: no\nCURB-65 baixo nao indica UTI; a estratificacao orienta enfermaria "
        "ou tratamento ambulatorial. [P2]",
    ),
]


# =============================================================================
# MODELOS DE DOCUMENTO INTERNO  (sinteticos)
# =============================================================================
# O item 1 do enunciado pede tres materiais para o fine-tuning: protocolos,
# perguntas frequentes E "modelos de laudos, receitas e procedimentos internos".
# Estes sao os terceiros.
#
# Eles ensinam ao modelo a ESTRUTURA dos documentos do hospital. Repare que a
# receita traz o campo de assinatura em branco: o modelo aprende que a
# prescricao termina com um medico assinando, e nao com ele proprio.
DOCUMENTOS = [
    {
        "id": "D1",
        "tipo": "laudo",
        "titulo": "Modelo de laudo de exame laboratorial",
        "texto": (
            "LAUDO LABORATORIAL - Hospital Vida Plena\n"
            "Identificacao: [PACIENTE] | Registro: [PRONTUARIO]\n"
            "Exame solicitado: <exame>. Metodo: <metodo>.\n"
            "Resultado: <valor> <unidade>. Referencia: <faixa>.\n"
            "Interpretacao: <achado, sem conduta>.\n"
            "Responsavel tecnico: <medico>, CRM <numero>."
        ),
    },
    {
        "id": "D2",
        "tipo": "receita",
        "titulo": "Modelo de prescricao medica",
        "texto": (
            "PRESCRICAO - Hospital Vida Plena\n"
            "Identificacao: [PACIENTE] | Registro: [PRONTUARIO]\n"
            "Alergias declaradas: <lista> (conferir antes de prescrever).\n"
            "1. <farmaco> <dose> <via> <intervalo> por <duracao>.\n"
            "Ajuste renal: verificar clearance antes da primeira dose.\n"
            "Assinatura do medico responsavel: ____________  CRM: ______"
        ),
    },
    {
        "id": "D3",
        "tipo": "procedimento",
        "titulo": "Modelo de descricao de procedimento",
        "texto": (
            "DESCRICAO DE PROCEDIMENTO - Hospital Vida Plena\n"
            "Procedimento: <nome>. Indicacao: <motivo>.\n"
            "Preparo: <jejum, consentimento, checagem de alergia>.\n"
            "Tecnica: <passos>. Intercorrencias: <descrever ou 'nenhuma'>.\n"
            "Orientacoes pos-procedimento: <cuidados>.\n"
            "Equipe responsavel: <nomes e CRMs>."
        ),
    },
]

# Perguntas sobre a estrutura desses documentos, no mesmo formato de resposta.
FAQ_DOCUMENTOS = [
    (
        "O que nao pode faltar numa prescricao do hospital?",
        "Decisao: yes\nAlergias declaradas conferidas antes de prescrever, dose, via, "
        "intervalo e duracao, e a assinatura do medico responsavel com CRM. [D2]",
    ),
    (
        "O laudo laboratorial deve trazer a conduta terapeutica?",
        "Decisao: no\nO laudo traz resultado, referencia e interpretacao do achado; "
        "a conduta e definida pelo medico assistente. [D1]",
    ),
    (
        "A descricao de procedimento precisa registrar intercorrencias?",
        "Decisao: yes\nO campo e obrigatorio e deve ser preenchido mesmo quando nao "
        "houve nenhuma. [D3]",
    ),
]


def anonimizar(texto: str) -> tuple[str, int]:
    """
    Remove identificadores diretos, preservando o dado clinico.

    O cuidado central: um anonimizador que apaga valor de exame entrega texto
    limpo e clinicamente inutil. Aqui so saem os padroes que identificam a
    pessoa - nome proprio apos marcador, CPF, telefone, numero de prontuario.
    """
    padroes = [
        (r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", "[CPF]"),
        (r"(?<!\d)\(?\d{2}\)?[\s-]?9?\d{4}-?\d{4}(?!\d)", "[TELEFONE]"),
        (r"\bprontu[aá]rio\s+n?[ºo°]?\s*\d+\b", "[PRONTUARIO]"),
        (r"\b(?:[Pp]aciente|[Ss]ra|[Ss]r|[Dd]ra|[Dd]r)\.?\s+"
         r"[A-Z][a-zà-ú]+(?:\s+[A-Z][a-zà-ú]+)+",
         "[PACIENTE]"),
    ]
    total = 0
    for padrao, marca in padroes:
        texto, n = re.subn(padrao, marca, texto)
        total += n
    return texto, total


def montar_exemplos_de_treino(n_pubmedqa: int = 350, semente: int = 42) -> list[dict]:
    """
    Monta o conjunto de treino no formato de conversa.

    Combina PubMedQA (evidencia cientifica real) com o material do hospital.
    O tamanho e deliberadamente pequeno: o objetivo do fine-tuning aqui e
    ensinar o FORMATO da resposta - decisao na primeira linha, fonte citada no
    fim -, e nao ensinar medicina ao modelo.
    """
    from datasets import load_dataset

    random.seed(semente)
    exemplos: list[dict] = []

    bruto = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    indices = random.sample(range(len(bruto)), min(n_pubmedqa, len(bruto)))

    for i in indices:
        item = bruto[i]
        contexto = " ".join(item["context"]["contexts"])[:1200]
        contexto, _ = anonimizar(contexto)
        exemplos.append({
            "pergunta": item["question"],
            "contexto": contexto,
            "resposta": (
                f"Decisao: {item['final_decision']}\n"
                f"{item['long_answer'][:400]} [E1]"
            ),
        })

    # O material do hospital entra repetido, para ter peso apesar de pequeno.
    fontes = {p["id"]: p for p in PROTOCOLOS} | {d["id"]: d for d in DOCUMENTOS}
    for _ in range(6):
        for pergunta, resposta in FAQ + FAQ_DOCUMENTOS:
            marcador = next(mid for mid in fontes if f"[{mid}]" in resposta)
            exemplos.append({
                "pergunta": pergunta,
                "contexto": f"[{marcador}] {fontes[marcador]['texto']}",
                "resposta": resposta,
            })

    random.shuffle(exemplos)
    return exemplos
