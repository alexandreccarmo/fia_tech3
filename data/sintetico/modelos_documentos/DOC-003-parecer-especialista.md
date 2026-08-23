---
id: DOC-003
titulo: Parecer de Especialista
tipo: parecer
setor_emissor: Corpo Clínico Especializado
campos_obrigatorios: [paciente, prontuario, data_emissao, especialidade, motivo_solicitacao, avaliacao, conduta_sugerida, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Parecer de Especialista é emitido quando a equipe assistente de um paciente internado ou em atendimento ambulatorial solicita a avaliação de um médico de outra especialidade para esclarecer dúvida diagnóstica, sugerir conduta terapêutica ou validar decisão clínica complexa. É comum em casos que envolvem múltiplos protocolos, como um paciente com lesão renal aguda (PROT-014) que necessita de avaliação de nefrologia, ou um caso de delirium (PROT-013) que requer parecer de neurologia ou psiquiatria. O documento é anexado ao prontuário e orienta, sem substituir, a decisão da equipe assistente principal, que permanece responsável pela condução do caso.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_emissao | Sim | Data e hora do parecer | {{data_emissao}} |
| especialidade | Sim | Especialidade médica consultada | {{especialidade_consultada}} |
| solicitante | Sim | Médico ou setor que solicitou o parecer | {{medico_solicitante}} |
| motivo_solicitacao | Sim | Motivo clínico da solicitação | {{motivo_solicitacao}} |
| avaliacao | Sim | Achados e análise do especialista | {{avaliacao_especialista}} |
| conduta_sugerida | Sim | Recomendações terapêuticas propostas | {{conduta_sugerida}} |
| responsavel | Sim | Nome do especialista responsável | {{nome_especialista}} |
| crm | Sim | Registro profissional do especialista | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Parecer de Especialista

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de emissão: {{data_emissao}}
Especialidade: {{especialidade_consultada}}
Solicitante: {{medico_solicitante}}

## Motivo da solicitação
{{motivo_solicitacao}}

## Avaliação
{{avaliacao_especialista}}

## Conduta sugerida
{{conduta_sugerida}}

## Responsável
Nome: {{nome_especialista}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Parecer de Especialista

Paciente: Rosalinda Ferruginosa Quatrocentos
Prontuário: PRT-10001
Data de emissão: 23/08/2026 09:15
Especialidade: Nefrologia
Solicitante: Dr. Anselmo Clínicageral Bagunçado, Clínica Médica

## Motivo da solicitação
Paciente internada com elevação de creatinina de 0,9 para 2,4 mg/dL em 48 horas, com diurese reduzida. Solicita-se avaliação de lesão renal aguda conforme PROT-014.

## Avaliação
Quadro compatível com lesão renal aguda estágio 2 (KDIGO), provavelmente de origem pré-renal associada a hipovolemia. Ausência de sinais de obstrução em exame de imagem prévio. Sem indicação de terapia renal substitutiva no momento.

## Conduta sugerida
Otimizar volemia com cristaloide balanceado, suspender anti-inflamatórios não esteroidais, ajustar dose de medicamentos de eliminação renal e reavaliar função renal em 24 horas.

## Responsável
Nome: Dra. Petúnia Rimalonga Cascavel
CRM: CRM/SP 000000
```

## 5. Regras de emissão
O parecer deve ser solicitado formalmente pela equipe assistente, com registro do motivo clínico específico da consulta, evitando solicitações genéricas sem pergunta objetiva a ser respondida. O especialista consultado tem prazo institucional de resposta conforme a gravidade do caso, sendo pareceres urgentes respondidos em até duas horas e os eletivos em até 24 horas. O parecer não transfere a responsabilidade pelo paciente ao especialista consultor, que atua em caráter consultivo; a equipe assistente principal decide sobre a incorporação ou não das recomendações ao plano terapêutico, registrando essa decisão em evolução própria. Todo parecer deve ser assinado eletronicamente pelo especialista responsável, com identificação de CRM, sendo vedada a emissão de recomendações terapêuticas sem essa assinatura. Em caso de divergência entre a conduta sugerida e a conduta adotada pela equipe assistente, deve haver justificativa registrada em prontuário.
