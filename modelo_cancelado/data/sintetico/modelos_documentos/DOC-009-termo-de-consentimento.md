---
id: DOC-009
titulo: Termo de Consentimento Livre e Esclarecido
tipo: termo_consentimento
setor_emissor: Enfermarias, UTI e Centro Cirúrgico
campos_obrigatorios: [paciente, prontuario, data_emissao, procedimento, riscos_informados, assinatura_paciente, assinatura_testemunha, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
O Termo de Consentimento Livre e Esclarecido é o documento pelo qual o paciente, ou seu responsável legal, declara ter recebido explicação clara sobre um procedimento diagnóstico ou terapêutico proposto, incluindo riscos, benefícios e alternativas, e autoriza formalmente sua realização. É exigido antes de procedimentos invasivos, como instalação de cateter venoso central, transfusões, terapia renal substitutiva em casos de lesão renal aguda (PROT-014) ou anticoagulação plena de alto risco (PROT-015). O documento protege tanto a autonomia do paciente quanto a equipe assistente, registrando que a decisão foi tomada de forma informada e voluntária.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_emissao | Sim | Data do esclarecimento e assinatura | {{data_emissao}} |
| procedimento | Sim | Procedimento proposto | {{nome_procedimento}} |
| riscos_informados | Sim | Riscos e benefícios explicados ao paciente | {{riscos_beneficios}} |
| alternativas | Não | Alternativas terapêuticas apresentadas | {{alternativas_terapeuticas}} |
| assinatura_paciente | Sim | Assinatura do paciente ou responsável legal | {{assinatura_paciente}} |
| assinatura_testemunha | Sim | Assinatura de testemunha do processo | {{assinatura_testemunha}} |
| responsavel | Sim | Médico que prestou o esclarecimento | {{nome_medico}} |
| crm | Sim | Registro profissional do responsável | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Termo de Consentimento Livre e Esclarecido

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de emissão: {{data_emissao}}
Procedimento proposto: {{nome_procedimento}}

## Riscos e benefícios informados
{{riscos_beneficios}}

## Alternativas apresentadas
{{alternativas_terapeuticas}}

## Declaração
Declaro ter sido informado(a) de forma clara sobre o procedimento acima, seus riscos e benefícios, e autorizo sua realização.

Assinatura do paciente/responsável legal: {{assinatura_paciente}}
Assinatura da testemunha: {{assinatura_testemunha}}

## Responsável pelo esclarecimento
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Termo de Consentimento Livre e Esclarecido

Paciente: Eufrosina Cateterina Almofariz
Prontuário: PRT-10001
Data de emissão: 23/08/2026
Procedimento proposto: Instalação de cateter venoso central em veia subclávia direita

## Riscos e benefícios informados
Explicados riscos de pneumotórax, sangramento local, infecção de corrente sanguínea e punção arterial acidental. Benefício de acesso venoso seguro para infusão de vasopressores conforme conduta prevista no PROT-001.

## Alternativas apresentadas
Foi discutida a possibilidade de acesso venoso periférico calibroso, considerada insuficiente diante da necessidade de infusão contínua de droga vasoativa.

## Declaração
Declaro ter sido informado(a) de forma clara sobre o procedimento acima, seus riscos e benefícios, e autorizo sua realização.

Assinatura do paciente/responsável legal: Eufrosina Cateterina Almofariz
Assinatura da testemunha: Rogério Testemunhoso Alcaparra

## Responsável pelo esclarecimento
Nome: Dr. Balduíno Veiacentral Trombonete
CRM: CRM/SP 000000
```

## 5. Regras de emissão
**O termo exige obrigatoriamente a assinatura do paciente ou de seu responsável legal**, sendo vedada a realização do procedimento com base apenas em consentimento verbal não registrado, exceto em situações de emergência com risco iminente de morte, quando a ausência de consentimento deve ser justificada por escrito. **É exigido também o registro de assinatura de testemunha**, que atesta que o processo de esclarecimento efetivamente ocorreu e que a assinatura do paciente é legítima. Quando o paciente estiver incapacitado de decidir, a assinatura deve ser colhida do responsável legal formalmente identificado, com registro do vínculo (cônjuge, filho, tutor legal). O médico responsável pelo esclarecimento deve assinar eletronicamente o documento, com CRM identificado, atestando que prestou as informações em linguagem acessível. Termos incompletos, sem as três assinaturas exigidas (paciente ou responsável, testemunha e médico), não autorizam a realização do procedimento.
