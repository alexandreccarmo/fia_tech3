---
id: DOC-002
titulo: Receita Médica
tipo: receita
setor_emissor: Ambulatório e Enfermarias
campos_obrigatorios: [paciente, prontuario, data_emissao, medicamentos, posologia, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
A Receita Médica é o documento pelo qual um médico do Hospital Vida Plena prescreve medicamentos a um paciente, definindo substância, dose, via de administração, frequência e duração do tratamento. É emitida ao final de consultas ambulatoriais, altas hospitalares (quando complementar ao Sumário de Alta) ou revisões de tratamento crônico, como no acompanhamento de hipertensão (PROT-005) ou diabetes tipo 2 (PROT-004). O documento é entregue ao paciente ou responsável legal e serve de base para dispensação em farmácia, sendo também anexado ao prontuário eletrônico.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_emissao | Sim | Data de emissão da receita | {{data_emissao}} |
| medicamentos | Sim | Lista de medicamentos prescritos | {{lista_medicamentos}} |
| posologia | Sim | Dose, via e frequência de cada item | {{posologia_detalhada}} |
| duracao_tratamento | Sim | Duração prevista do tratamento | {{duracao_tratamento}} |
| observacoes | Não | Orientações adicionais ao paciente | {{observacoes_gerais}} |
| responsavel | Sim | Nome do médico prescritor | {{nome_medico}} |
| crm | Sim | Registro profissional do prescritor | {{crm_responsavel}} |
| status_validacao | Sim | Situação da validação humana da prescrição | {{status_validacao}} |

## 3. Modelo (gabarito)

```markdown
# Receita Médica

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de emissão: {{data_emissao}}

## Medicamentos prescritos
{{lista_medicamentos}}

## Posologia
{{posologia_detalhada}}

## Duração do tratamento
{{duracao_tratamento}}

## Observações
{{observacoes_gerais}}

## Prescritor
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
Status de validação: {{status_validacao}}
```

## 4. Exemplo preenchido

```markdown
# Receita Médica

Paciente: Firmino Datavelha Alcachofra
Prontuário: PRT-10001
Data de emissão: 23/08/2026

## Medicamentos prescritos
1. Losartana potássica 50 mg
2. Hidroclorotiazida 25 mg

## Posologia
1. Losartana potássica 50 mg — 1 comprimido, via oral, 1 vez ao dia, pela manhã
2. Hidroclorotiazida 25 mg — 1 comprimido, via oral, 1 vez ao dia, pela manhã

## Duração do tratamento
Uso contínuo, com reavaliação em 30 dias conforme PROT-005 (Hipertensão)

## Observações
Monitorar pressão arterial em casa duas vezes por semana e retornar antes do prazo em caso de tontura ou hipotensão

## Prescritor
Nome: Dra. Ceumira Prontuária Xisto
CRM: CRM/SP 000000
Status de validação: validado por médico responsável em 23/08/2026
```

## 5. Regras de emissão
**Toda prescrição precisa ser assinada por médico responsável, identificado por CRM.** Nenhuma receita é considerada válida sem essa assinatura, mesmo quando o rascunho tenha sido preparado por outra pessoa ou sistema. **Nenhum sistema automatizado ou assistente de inteligência artificial pode emitir prescrição sem validação humana registrada.** Ferramentas de apoio, incluindo assistentes de IA integrados ao prontuário eletrônico do Hospital Vida Plena, **podem apenas preparar rascunho, que fica pendente até a validação** explícita de um médico habilitado, que deve revisar dose, via, interações medicamentosas e adequação ao quadro clínico antes de assinar. O campo `status_validacao` deve refletir sempre esse estado: "rascunho pendente" ou "validado por médico responsável", nunca podendo ser emitido em nome do paciente enquanto pendente. Receitas de medicamentos controlados seguem numeração sequencial própria e talão específico, conforme legislação sanitária vigente. Qualquer alteração após a emissão exige nova receita, sendo vedado rasurar ou reescrever documento já assinado.
