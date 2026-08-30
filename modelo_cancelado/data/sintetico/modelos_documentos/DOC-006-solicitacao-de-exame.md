---
id: DOC-006
titulo: Solicitação de Exame
tipo: solicitacao_exame
setor_emissor: Ambulatório, Pronto-Socorro e Enfermarias
campos_obrigatorios: [paciente, prontuario, data_emissao, exames_solicitados, hipotese_diagnostica, prioridade, responsavel, crm]
exige_assinatura_medica: true
natureza: sintético
---

## Aviso
> Documento sintético, gerado para fins acadêmicos no âmbito do Tech Challenge Fase 3 (8IADT). O Hospital Vida Plena é fictício. Este material não constitui documento médico real.

## 1. Finalidade
A Solicitação de Exame é o documento pelo qual um médico do Hospital Vida Plena requisita exames laboratoriais, de imagem ou funcionais necessários para investigação diagnóstica ou acompanhamento terapêutico de um paciente. É emitida em qualquer ponto do atendimento, desde a triagem no Pronto-Socorro até o seguimento ambulatorial de doenças crônicas, sendo frequentemente vinculada a protocolos clínicos, como a solicitação de troponina e eletrocardiograma prevista no PROT-002 (Dor Torácica) ou de tomografia de crânio no PROT-003 (AVC Agudo). O documento direciona o fluxo do paciente até o setor de diagnóstico correspondente e determina a prioridade de realização do exame.

## 2. Estrutura do documento

| Campo | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| paciente | Sim | Nome completo do paciente | {{nome_paciente}} |
| prontuario | Sim | Número do prontuário hospitalar | {{prontuario}} |
| data_emissao | Sim | Data e hora da solicitação | {{data_emissao}} |
| setor_solicitante | Sim | Setor que originou a solicitação | {{setor_solicitante}} |
| exames_solicitados | Sim | Lista de exames requisitados | {{lista_exames}} |
| hipotese_diagnostica | Sim | Hipótese diagnóstica que justifica o exame | {{hipotese_diagnostica}} |
| prioridade | Sim | Urgência da realização (rotina, urgente, emergência) | {{nivel_prioridade}} |
| observacoes | Não | Informações adicionais para o setor executor | {{observacoes_exame}} |
| responsavel | Sim | Médico solicitante | {{nome_medico}} |
| crm | Sim | Registro profissional do solicitante | {{crm_responsavel}} |

## 3. Modelo (gabarito)

```markdown
# Solicitação de Exame

Paciente: {{nome_paciente}}
Prontuário: {{prontuario}}
Data de emissão: {{data_emissao}}
Setor solicitante: {{setor_solicitante}}

## Exames solicitados
{{lista_exames}}

## Hipótese diagnóstica
{{hipotese_diagnostica}}

## Prioridade
{{nivel_prioridade}}

## Observações
{{observacoes_exame}}

## Solicitante
Nome: {{nome_medico}}
CRM: {{crm_responsavel}}
```

## 4. Exemplo preenchido

```markdown
# Solicitação de Exame

Paciente: Nicanor Batimentário Guaraná
Prontuário: PRT-10001
Data de emissão: 23/08/2026 22:10
Setor solicitante: Pronto-Socorro

## Exames solicitados
1. Eletrocardiograma de 12 derivações
2. Troponina de alta sensibilidade (seriada em 0h e 3h)
3. Radiografia de tórax

## Hipótese diagnóstica
Dor torácica atípica em paciente com fatores de risco cardiovascular, investigação de síndrome coronariana aguda conforme PROT-002.

## Prioridade
Emergência

## Observações
Paciente com histórico de hipertensão e tabagismo, sinais vitais estáveis no momento da solicitação.

## Solicitante
Nome: Dr. Heráclito Ausculta Pimentão
CRM: CRM/SP 000000
```

## 5. Regras de emissão
A solicitação deve conter hipótese diagnóstica clara, evitando pedidos genéricos sem justificativa clínica, para permitir a correta priorização pelo setor executor. A classificação de prioridade deve seguir critérios institucionais: "emergência" para exames que alteram conduta imediata, "urgente" para resultados necessários em até 24 horas e "rotina" para investigação eletiva. Exames de imagem com uso de contraste exigem confirmação de função renal e ausência de alergia prévia registrada na própria solicitação. Toda solicitação deve ser assinada eletronicamente pelo médico responsável, com CRM identificado, sendo vedado o encaminhamento de exames sem essa autorização. Cancelamentos ou alterações de exames já solicitados devem ser registrados como nova solicitação com justificativa, mantendo o registro original para fins de auditoria.
