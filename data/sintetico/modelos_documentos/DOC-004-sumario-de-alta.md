---
id: DOC-004
titulo: Sumário de Alta
tipo: sumario_alta
setor_emissor: Enfermarias e UTI
campos_obrigatorios: [paciente, prontuario, data_internacao, data_alta, diagnostico_principal, resumo_evolucao, condicoes_alta, orientacoes, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Sumário de Alta é o documento que consolida toda a internação do paciente no Hospital Vida Plena, sendo emitido no momento da alta hospitalar, seja por melhora clínica, transferência para outro serviço ou óbito. Ele resume o motivo da internação, a evolução clínica, os exames relevantes, o tratamento realizado e as condições do paciente no momento da saída. É essencial para garantir a continuidade do cuidado, servindo de referência para o médico da atenção primária, para o ambulatório de seguimento e para reinternações futuras. Casos conduzidos por protocolos como PROT-010 (Insuficiência Cardíaca) ou PROT-011 (DPOC Exacerbado) costumam referenciar diretamente o protocolo seguido durante a internação.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_internacao | Sim | Data de admissão hospitalar | {{data_internacao}} |
| data_alta | Sim | Data da alta hospitalar | {{data_alta}} |
| diagnostico_principal | Sim | Diagnóstico principal da internação | {{diagnostico_principal}} |
| diagnosticos_secundarios | Não | Comorbidades relevantes | {{diagnosticos_secundarios}} |
| resumo_evolucao | Sim | Resumo da evolução clínica | {{resumo_evolucao}} |
| condicoes_alta | Sim | Estado clínico no momento da alta | {{condicoes_alta}} |
| orientacoes | Sim | Orientações e medicamentos de alta | {{orientacoes_alta}} |
| responsavel | Sim | Médico responsável pela alta | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Sumário de Alta

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de internação: {{data_internacao}}
Data de alta: {{data_alta}}

## Diagnóstico principal
{{diagnostico_principal}}

## Diagnósticos secundários
{{diagnosticos_secundarios}}

## Resumo da evolução
{{resumo_evolucao}}

## Condições de alta
{{condicoes_alta}}

## Orientações
{{orientacoes_alta}}

## Responsável
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Sumário de Alta

Paciente: Wanderley Prontuarindo Sabugosa
Prontuário: PRT-10001
Data de internação: 15/08/2026
Data de alta: 23/08/2026

## Diagnóstico principal
Insuficiência cardíaca descompensada, classe funcional III (NYHA)

## Diagnósticos secundários
Hipertensão arterial sistêmica, fibrilação atrial crônica

## Resumo da evolução
Paciente admitido com dispneia e edema de membros inferiores, conduzido conforme PROT-010, com resposta favorável a diuréticos endovenosos e otimização de terapia com inibidor da enzima conversora. Evoluiu com melhora progressiva do quadro congestivo ao longo de sete dias.

## Condições de alta
Paciente eupneico em repouso, sem edema, sinais vitais estáveis, deambulando com autonomia.

## Orientações
Manter furosemida 40 mg via oral pela manhã, restrição hídrica de 1,5 litro por dia, retorno ambulatorial em 7 dias e pesagem diária em domicílio.

## Responsável
Nome: Dr. Osvaldino Cardiopulso Retumbante
CRM: CRM/SP 000000
```

## 5. Regras de emissão
O Sumário de Alta deve ser concluído antes da liberação efetiva do paciente, contendo obrigatoriamente diagnóstico principal, resumo cronológico da evolução e orientações claras de continuidade do cuidado. Nenhuma alta é considerada completa sem esse documento assinado eletronicamente pelo médico responsável, com CRM identificado. O documento deve mencionar explicitamente os protocolos clínicos internos seguidos durante a internação, quando aplicável, para permitir auditoria e rastreabilidade das condutas. Em caso de transferência para outro serviço, o sumário deve ser entregue junto à guia de transferência. Em caso de óbito, o sumário substitui parcialmente a declaração de óbito, mas não a dispensa, devendo ambos os documentos ser consistentes entre si. Correções após a emissão exigem sumário complementar, preservando o documento original no histórico do prontuário.
