---
id: DOC-008
titulo: Relatório de Intercorrência
tipo: relatorio_intercorrencia
setor_emissor: Enfermarias e UTI
campos_obrigatorios: [paciente, prontuario, data_hora_evento, tipo_intercorrencia, descricao_evento, conduta_imediata, desfecho, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Relatório de Intercorrência é emitido sempre que ocorre um evento clínico inesperado ou agudo durante a internação do paciente, como piora súbita do quadro, evento adverso relacionado a medicamento, queda, parada cardiorrespiratória ou reação alérgica. O documento registra detalhadamente o que ocorreu, quando, quem foi acionado e qual conduta foi tomada, sendo essencial tanto para a continuidade do cuidado quanto para os processos institucionais de gestão de risco e segurança do paciente. Intercorrências como sangramento digestivo agudo podem acionar diretamente o PROT-012 (Hemorragia Digestiva), e episódios de confusão mental aguda podem remeter ao PROT-013 (Delirium).

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_hora_evento | Sim | Data e hora exata do evento | {{data_hora_evento}} |
| local_evento | Sim | Local onde ocorreu a intercorrência | {{local_evento}} |
| tipo_intercorrencia | Sim | Categoria do evento (queda, reação adversa etc.) | {{tipo_intercorrencia}} |
| descricao_evento | Sim | Descrição cronológica do ocorrido | {{descricao_evento}} |
| conduta_imediata | Sim | Ações tomadas no momento do evento | {{conduta_imediata}} |
| desfecho | Sim | Resultado clínico após a intervenção | {{desfecho_evento}} |
| responsavel | Sim | Médico que registra o relatório | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Relatório de Intercorrência

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data e hora do evento: {{data_hora_evento}}
Local: {{local_evento}}
Tipo de intercorrência: {{tipo_intercorrencia}}

## Descrição do evento
{{descricao_evento}}

## Conduta imediata
{{conduta_imediata}}

## Desfecho
{{desfecho_evento}}

## Responsável
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Relatório de Intercorrência

Paciente: Bartolomeu Vertiginoso Quebracabeça
Prontuário: PRT-10001
Data e hora do evento: 23/08/2026 03:20
Local: Enfermaria de Clínica Médica, leito 214
Tipo de intercorrência: Queda do leito

## Descrição do evento
Paciente encontrado no chão ao lado do leito pela equipe de enfermagem durante ronda noturna, consciente, orientado, referindo tontura ao tentar levantar-se sozinho para ir ao banheiro. Grades de proteção estavam abaixadas.

## Conduta imediata
Realizada avaliação neurológica e de sinais vitais, sem alterações significativas. Solicitada radiografia de bacia e crânio para descartar fratura, ambas sem alterações. Grades de proteção elevadas e campainha reposicionada ao alcance do paciente.

## Desfecho
Paciente sem lesões identificadas, mantido em observação por 24 horas com aferição de sinais vitais a cada 4 horas. Evento notificado ao núcleo de segurança do paciente.

## Responsável
Nome: Dra. Serafina Notívaga Coelhotardio
CRM: CRM/SP 000000
```

## 5. Regras de emissão
O Relatório de Intercorrência deve ser preenchido imediatamente após a estabilização do paciente, nunca ultrapassando duas horas do evento, para preservar a precisão dos detalhes registrados. Todo evento classificado como grave, incluindo parada cardiorrespiratória, queda com lesão ou erro de medicação, deve ser adicionalmente notificado ao núcleo de segurança do paciente do Hospital Vida Plena, independentemente do registro em prontuário. A descrição deve ser factual e cronológica, evitando juízos de valor sobre a conduta de terceiros, sendo eventuais falhas de processo tratadas em fluxo de gestão de risco separado. O relatório deve ser assinado eletronicamente pelo médico responsável presente no momento do evento ou que assumiu a condução do caso, com CRM identificado. Relatórios de intercorrência não substituem a evolução diária, devendo ambos os registros coexistir no prontuário do paciente.
