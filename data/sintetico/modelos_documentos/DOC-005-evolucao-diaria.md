---
id: DOC-005
titulo: Evolução Diária
tipo: evolucao
setor_emissor: Enfermarias e UTI
campos_obrigatorios: [paciente, prontuario, data_hora, dia_internacao, estado_geral, exame_fisico, avaliacao, conduta, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
A Evolução Diária é o registro médico produzido pelo menos uma vez por período de internação, documentando o estado clínico do paciente, os achados do exame físico, a interpretação da equipe assistente e a conduta adotada naquele momento. É o principal instrumento de acompanhamento longitudinal do paciente internado, permitindo reconstruir a linha do tempo clínica em casos complexos, como acompanhamento de anticoagulação (PROT-015) ou monitoramento de tromboembolismo (PROT-008). O documento é registrado em prontuário eletrônico a cada visita médica relevante, incluindo intercorrências, e serve de base para decisões de toda a equipe multiprofissional.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_hora | Sim | Data e hora do registro | {{data_hora_evolucao}} |
| dia_internacao | Sim | Dia de internação correspondente | {{dia_internacao}} |
| estado_geral | Sim | Estado geral e queixas do paciente | {{estado_geral}} |
| sinais_vitais | Sim | Sinais vitais aferidos no período | {{sinais_vitais}} |
| exame_fisico | Sim | Achados relevantes do exame físico | {{exame_fisico}} |
| avaliacao | Sim | Interpretação clínica do quadro | {{avaliacao_clinica}} |
| conduta | Sim | Conduta e ajustes terapêuticos | {{conduta_adotada}} |
| responsavel | Sim | Médico responsável pelo registro | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Evolução Diária

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data e hora: {{data_hora_evolucao}}
Dia de internação: {{dia_internacao}}

## Estado geral
{{estado_geral}}

## Sinais vitais
{{sinais_vitais}}

## Exame físico
{{exame_fisico}}

## Avaliação
{{avaliacao_clinica}}

## Conduta
{{conduta_adotada}}

## Responsável
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Evolução Diária

Paciente: Filomeno Cateterzinho Alvorada
Prontuário: PRT-10001
Data e hora: 23/08/2026 07:40
Dia de internação: 4º dia

## Estado geral
Paciente refere melhora da dor no membro inferior esquerdo, nega dispneia, aceita dieta via oral integralmente.

## Sinais vitais
PA 118/76 mmHg, FC 78 bpm, FR 16 irpm, SatO2 97% em ar ambiente, Tax 36,5°C

## Exame físico
Membro inferior esquerdo com edema em regressão, panturrilha menos dolorosa à palpação, ausculta pulmonar sem ruídos adventícios, abdome normal.

## Avaliação
Evolução favorável de trombose venosa profunda em tratamento conforme PROT-008, sem sinais de sangramento ou embolização pulmonar.

## Conduta
Manter anticoagulação plena conforme PROT-015, solicitar novo controle de coagulograma em 48 horas, iniciar deambulação assistida com meia elástica de compressão.

## Responsável
Nome: Dra. Vitalina Coagulete Marimbondo
CRM: CRM/SP 000000
```

## 5. Regras de emissão
Toda evolução deve ser registrada em ordem cronológica, sem exclusão de registros anteriores, garantindo rastreabilidade completa da internação. É obrigatório o registro de pelo menos uma evolução médica por dia de internação em enfermaria, e a cada seis a oito horas em unidades de terapia intensiva, ou sempre que houver intercorrência relevante. O texto deve ser objetivo, evitando abreviações ambíguas, e deve sempre concluir com a conduta adotada, ainda que seja "manter conduta atual". Toda evolução deve ser assinada eletronicamente pelo profissional responsável, com identificação de CRM, sendo vedado o registro em nome de terceiros. Alterações após o fechamento do registro exigem nova evolução complementar, nunca a edição do texto original, preservando a integridade do histórico clínico do paciente.
