---
id: DOC-007
titulo: Encaminhamento Ambulatorial
tipo: encaminhamento
setor_emissor: Ambulatório e Enfermarias
campos_obrigatorios: [paciente, prontuario, data_emissao, especialidade_destino, motivo_encaminhamento, resumo_clinico, prioridade, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Encaminhamento Ambulatorial é emitido quando o médico assistente identifica a necessidade de seguimento do paciente em outra especialidade ou serviço, seja para dar continuidade a um tratamento iniciado durante a internação, seja para investigação eletiva que não requer estrutura hospitalar. É comumente utilizado na alta de pacientes que seguiram protocolos como PROT-004 (Diabetes Tipo 2), com encaminhamento para endocrinologia, ou PROT-013 (Delirium), com encaminhamento para neurologia ou geriatria após a resolução do quadro agudo. O documento orienta a recepção do ambulatório na priorização da agenda e garante que informações clínicas relevantes acompanhem o paciente.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_emissao | Sim | Data de emissão do encaminhamento | {{data_emissao}} |
| especialidade_destino | Sim | Especialidade ou serviço de destino | {{especialidade_destino}} |
| motivo_encaminhamento | Sim | Motivo clínico do encaminhamento | {{motivo_encaminhamento}} |
| resumo_clinico | Sim | Resumo do quadro e histórico relevante | {{resumo_clinico}} |
| prioridade | Sim | Urgência do agendamento | {{nivel_prioridade}} |
| responsavel | Sim | Médico que emite o encaminhamento | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Encaminhamento Ambulatorial

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de emissão: {{data_emissao}}
Especialidade de destino: {{especialidade_destino}}

## Motivo do encaminhamento
{{motivo_encaminhamento}}

## Resumo clínico
{{resumo_clinico}}

## Prioridade
{{nivel_prioridade}}

## Responsável
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Encaminhamento Ambulatorial

Paciente: Isaltina Glicemita Cravovermelho
Prontuário: PRT-10001
Data de emissão: 23/08/2026
Especialidade de destino: Endocrinologia

## Motivo do encaminhamento
Paciente com diagnóstico recente de diabetes tipo 2 durante internação, necessitando ajuste fino de terapia hipoglicemiante e orientação especializada de longo prazo.

## Resumo clínico
Internação conduzida conforme PROT-004, com hemoglobina glicada de 11,2% na admissão e glicemias capilares elevadas mesmo após ajuste de insulina basal. Recebeu alta com insulina NPH e metformina, necessitando acompanhamento próximo.

## Prioridade
Prioritário (até 15 dias)

## Responsável
Nome: Dr. Gumercindo Metabolizado Feijão
CRM: CRM/SP 000000
```

## 5. Regras de emissão
Todo encaminhamento deve conter resumo clínico suficiente para que o especialista de destino compreenda o caso sem necessidade de acesso imediato ao prontuário completo, incluindo diagnósticos relevantes, exames pertinentes e tratamento em curso. A prioridade deve ser classificada como "emergência" (atendimento em até 24 horas), "prioritário" (até 15 dias) ou "eletivo" (conforme disponibilidade de agenda), de acordo com critérios institucionais de risco. Encaminhamentos emitidos na alta hospitalar devem ser entregues ao paciente junto ao Sumário de Alta e à Prescrição de Alta, quando aplicável. O documento deve ser assinado eletronicamente pelo médico responsável, com CRM identificado, sendo vedada a emissão de encaminhamento sem essa validação. Encaminhamentos não utilizados pelo paciente em prazo superior a 90 dias perdem validade e devem ser reemitidos mediante nova avaliação clínica.
