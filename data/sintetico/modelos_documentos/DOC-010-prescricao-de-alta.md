---
id: DOC-010
titulo: Prescrição de Alta
tipo: prescricao_alta
setor_emissor: Enfermarias e UTI
campos_obrigatorios: [paciente, prontuario, data_alta, medicamentos, posologia, duracao_tratamento, sinais_alerta, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
A Prescrição de Alta é o documento que define o tratamento medicamentoso a ser mantido pelo paciente após deixar o Hospital Vida Plena, consolidando os ajustes terapêuticos feitos durante a internação em um plano claro para uso domiciliar. É emitida junto ao Sumário de Alta, garantindo que o paciente saia da instituição com orientação precisa sobre quais medicamentos manter, suspender ou iniciar, como ocorre tipicamente ao final do tratamento de infecção do trato urinário (PROT-006) ou de pneumonia comunitária (PROT-007). O documento também informa sinais de alerta que justificam retorno imediato ao serviço de saúde.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_alta | Sim | Data da alta hospitalar | {{data_alta}} |
| medicamentos | Sim | Lista de medicamentos de uso domiciliar | {{lista_medicamentos}} |
| posologia | Sim | Dose, via e frequência de cada item | {{posologia_detalhada}} |
| duracao_tratamento | Sim | Duração de cada medicamento | {{duracao_tratamento}} |
| sinais_alerta | Sim | Sinais que exigem retorno imediato | {{sinais_alerta}} |
| retorno_ambulatorial | Não | Data ou prazo de retorno agendado | {{retorno_ambulatorial}} |
| responsavel | Sim | Médico responsável pela alta | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |
| status_validacao | Sim | Situação da validação humana da prescrição | {{status_validacao}} |

## 3. Modelo (gabarito)

```markdown
# Prescrição de Alta

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de alta: {{data_alta}}

## Medicamentos de uso domiciliar
{{lista_medicamentos}}

## Posologia
{{posologia_detalhada}}

## Duração do tratamento
{{duracao_tratamento}}

## Sinais de alerta para retorno imediato
{{sinais_alerta}}

## Retorno ambulatorial
{{retorno_ambulatorial}}

## Prescritor
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
Status de validação: {{status_validacao}}
```

## 4. Exemplo preenchido

```markdown
# Prescrição de Alta

Paciente: Higínio Broncopneumônico Saracura
Prontuário: PRT-10001
Data de alta: 23/08/2026

## Medicamentos de uso domiciliar
1. Amoxicilina + clavulanato 875/125 mg
2. Paracetamol 750 mg (se dor ou febre)

## Posologia
1. Amoxicilina + clavulanato 875/125 mg — 1 comprimido, via oral, a cada 12 horas
2. Paracetamol 750 mg — 1 comprimido, via oral, até de 6 em 6 horas se necessário

## Duração do tratamento
Antibiótico por mais 5 dias, completando 7 dias de tratamento conforme PROT-007

## Sinais de alerta para retorno imediato
Falta de ar, febre persistente após 48 horas de antibiótico, dor torácica intensa ou confusão mental

## Retorno ambulatorial
Retorno em 7 dias no ambulatório de pneumologia

## Prescritor
Nome: Dra. Marcelina Pulmonária Girassol
CRM: CRM/SP 000000
Status de validação: validado por médico responsável em 23/08/2026
```

## 5. Regras de emissão
**Toda prescrição precisa ser assinada por médico responsável, identificado por CRM.** A Prescrição de Alta somente pode ser entregue ao paciente após essa assinatura eletrônica formal, nunca antes. **Nenhum sistema automatizado ou assistente de inteligência artificial pode emitir prescrição sem validação humana registrada**, mesmo quando o conteúdo for gerado a partir de sugestões baseadas nos protocolos clínicos internos do Hospital Vida Plena. Ferramentas de apoio à decisão, incluindo assistentes de IA, **podem apenas preparar rascunho, que fica pendente até a validação** de um médico habilitado, que deve conferir doses, interações e adequação ao quadro de alta antes de assinar. O campo `status_validacao` deve indicar claramente "rascunho pendente" ou "validado por médico responsável". A prescrição deve obrigatoriamente listar sinais de alerta compreensíveis ao paciente leigo, e sempre que possível indicar prazo de retorno ambulatorial, evitando alta sem plano de continuidade do cuidado.
