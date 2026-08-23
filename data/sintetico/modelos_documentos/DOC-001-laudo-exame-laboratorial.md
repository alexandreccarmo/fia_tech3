---
id: DOC-001
titulo: Laudo de Exame Laboratorial
tipo: laudo
setor_emissor: Laboratório Central
campos_obrigatorios: [paciente, prontuario, data_emissao, exame, resultado, interpretacao, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Laudo de Exame Laboratorial é o documento emitido pelo Laboratório Central do Hospital Vida Plena para registrar formalmente os resultados de exames de sangue, urina, líquidos corporais ou culturas microbiológicas solicitados durante o atendimento de um paciente. Ele é emitido sempre que uma amostra biológica é processada e analisada, seja em contexto ambulatorial, de internação ou de emergência, e serve como base objetiva para decisões diagnósticas e terapêuticas registradas em prontuário. O laudo pode ser referenciado por protocolos clínicos internos, como o PROT-001 (Sepse e Choque Séptico), que exige dosagem de lactato, ou o PROT-009 (Cetoacidose Diabética), que depende de gasometria e glicemia seriada. Após emissão, o laudo é anexado ao prontuário eletrônico do paciente e disponibilizado à equipe assistencial responsável.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_nascimento | Sim | Data de nascimento do paciente | {{data_nascimento}} |
| data_emissao | Sim | Data e hora da emissão do laudo | {{data_emissao}} |
| setor_solicitante | Sim | Unidade que solicitou o exame | {{setor_solicitante}} |
| exame | Sim | Nome do exame realizado | {{nome_exame}} |
| metodo | Não | Método analítico utilizado | {{metodo_analitico}} |
| resultado | Sim | Valores obtidos e unidades de referência | {{resultado_exame}} |
| interpretacao | Sim | Interpretação clínica do resultado | {{interpretacao_clinica}} |
| responsavel | Sim | Nome do responsável técnico ou médico | {{nome_responsavel}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Laudo de Exame Laboratorial

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de nascimento: {{data_nascimento}}
Setor solicitante: {{setor_solicitante}}
Data de emissão: {{data_emissao}}

## Exame
Nome do exame: {{nome_exame}}
Método: {{metodo_analitico}}

## Resultado
{{resultado_exame}}

## Interpretação clínica
{{interpretacao_clinica}}

## Responsável técnico
Nome: {{nome_responsavel}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Laudo de Exame Laboratorial

Paciente: Aurora Estrelabranca Nogueira
Prontuário: PRT-10001
Data de nascimento: 01/01/1900
Setor solicitante: Pronto-Socorro
Data de emissão: 23/08/2026 14:32

## Exame
Nome do exame: Lactato sérico
Método: Enzimático colorimétrico

## Resultado
Lactato: 4,8 mmol/L (referência: 0,5 a 2,0 mmol/L)

## Interpretação clínica
Hiperlactatemia significativa, compatível com hipoperfusão tecidual. Sugere-se correlação com quadro clínico de suspeita de sepse (ver PROT-001) e reavaliação seriada em 2 horas conforme meta de pacote de 1 hora.

## Responsável técnico
Nome: Dr. Belisário Quimiotesta Vãozinho
CRM: CRM/SP 000000
```

## 5. Regras de emissão
O laudo somente pode ser liberado após validação técnica do responsável pelo setor de análises clínicas, que confere a coerência analítica dos valores antes da assinatura eletrônica. Resultados classificados como "críticos" pelo sistema laboratorial (por exemplo, potássio sérico acima de 6,5 mEq/L ou lactato acima de 4 mmol/L) devem ser comunicados verbalmente à equipe assistencial no momento da liberação, além do registro escrito, conforme rotina de valores de pânico do Hospital Vida Plena. Nenhum laudo pode ser emitido sem identificação completa do paciente e do prontuário, sob pena de reprocessamento da amostra. Alterações posteriores à liberação exigem emissão de laudo complementar ou retificado, nunca a edição do documento original, preservando o histórico auditável. Todo laudo deve conter a assinatura eletrônica do responsável técnico com respectivo CRM, sendo vedada a liberação de resultados sem essa validação profissional.
