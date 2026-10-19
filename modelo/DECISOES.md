# Decisões do modelo e por quê

Registro das discussões que levaram ao modelo atual. Serve para o relatório, para o artigo e para quem entrar no projeto depois.

---

## 1. Por que MicroPython, e não C/TFLite INT8

A proposta e o roadmap falavam em C/C++ com TinyML (TensorFlow Lite Micro, rede quantizada em INT8). O firmware do projeto será em MicroPython, então trocamos.

- A rede é muito pequena (24-16-3, 451 parâmetros), e a conta dela cabe em poucas linhas de Python: normalizar, multiplicar e somar, ReLU e escolher a maior saída. Não precisa de biblioteca de inferência.
- Em float32, os pesos ocupam cerca de 2 KB em `array('f')`. A quantização INT8 economizaria memória que não nos falta.
- A classificação roda uma vez por dia, então a velocidade não é problema mesmo sem FPU.
- Conferido: o `classificador.py`, rodando num MicroPython compilado com float de 32 bits como o do RP2040, dá a mesma classe que o Keras em 100% dos dias de teste.
- **Falta:** medir o tempo de inferência na BitDogLab com o `teste_classificador.py`.

**Para o relatório:** justificar a troca com o Prof. Max, porque ela muda o que a proposta prometia ("TinyML em C" virou "MLP embarcada em MicroPython").

---

## 2. Por que há constantes no topo do `01_gerar_dataset.py`

A pergunta foi: "não deveria ser tudo calculado?".

A maior parte já é calculada: TAW, RAW, o limite do crítico, o Kc de cada dia, a ETc, o Dr diário e as janelas de 3, 7 e 14 dias. Ficam como constantes só os **dados de entrada**, que nenhuma conta tira do clima:

| Constante | Por que não é calculada |
|---|---|
| Água disponível do solo (θCC − θPMP) | Depende do tipo de solo; vem de análise de laboratório ou de tabela (FAO-56, tab. 19). |
| Profundidade da raiz | Mede-se no campo ou pega-se da literatura. |
| `P_DEPLECAO` = 0,35 | É da cultura (uva de mesa), tabelado no FAO-56 (tab. 22). |
| `KS_CRITICO` = 0,5 | É uma definição nossa de onde começa o "crítico". |
| `CICLO_DIAS` = 109 | Vem de Ávila Netto et al. (2000) para Petrolina. |
| Data da poda | Depende do calendário de cada produtor. |

O ponto fraco não é serem constantes, e sim que algumas foram **assumidas** (marcadas `ASSUMIDO` no código) e precisam ser confirmadas com o Prof. Max.

---

## 3. O que é TAW

**TAW** (*Total Available Water*, água total disponível) é quanta água o solo consegue guardar ao alcance das raízes, em mm.

- Capacidade de campo (θCC): o solo "cheio", depois que o excesso de uma chuva forte escorreu.
- Ponto de murcha (θPMP): o solo tão seco que a raiz não consegue puxar água.
- A diferença entre os dois é a água que a planta consegue usar.

`TAW = 1000 × (θCC − θPMP) × Zr`. Exemplo: `1000 × 0,11 × 0,6 = 66 mm`.

Os limites das classes saem dele:
- **RAW = 0,35 × TAW**: até gastar isso, a planta não sofre. É o limite do "adequado".
- **Dr crítico = TAW − 0,5 × (1 − 0,35) × TAW**: a partir daqui, a planta transpira menos da metade do normal. É o limite do "crítico".

O `Dr` que o balanço acompanha é quanto desse "tanque" já foi gasto: 0 é cheio, TAW é seco.

---

## 4. Petrolina e Juazeiro não têm o mesmo solo

Pela literatura geral da região (a confirmar com a Embrapa Semiárido e o Prof. Max):

- **Petrolina** (perímetros Nilo Coelho, Bebedouro): quase tudo solo arenoso ou franco-arenoso (Argissolos, Latossolos, Neossolos Quartzarênicos), com cerca de 0,06 a 0,12 m³/m³ de água disponível.
- **Juazeiro** (Maniçoba, Mandacaru): há áreas de **Vertissolo**, argiloso, que segura bem mais água.
- **Raiz da videira com gotejamento:** concentrada nos primeiros 40 a 60 cm.
- **Poda:** não há inverno, então cada produtor poda o ano inteiro, escalonando a colheita (cerca de 2,5 safras por ano). As datas fixas de 15/01 e 15/07 da primeira versão não representavam o Vale.

Conclusão: o TAW pode variar muito de uma margem do rio para a outra, então os parâmetros do solo precisam ser configuráveis.

---

## 5. Configurar só no firmware não bastava

A primeira versão aprendeu as fronteiras entre as classes para **um** solo, o de 66 mm. Ela não recebia o TAW como entrada. Num solo arenoso de 40 mm, os mesmos 7 dias sem chuva já seriam críticos, e ela não saberia. Trocar a constante só na placa não mudaria nada.

Havia dois caminhos:

1. **Um modelo por local:** a configuração fica no PC, e cada local novo exige rodar 01 → 02 → 03 e gravar novos pesos.
2. **Um modelo geral:** a simulação sorteia solos e raízes, o TAW entra como feature, e a configuração fica na placa. O mesmo arquivo de pesos serve em qualquer lugar.

---

## 6. O teste que decidiu

Geramos dois conjuntos de dados, ambos com a poda em qualquer dia do ano:
- **Solo fixo:** 66 mm.
- **Solos variados:** água disponível de 0,06 a 0,18 m³/m³ e raiz de 0,4 a 0,8 m, ou seja, TAW de 25 a 140 mm.

Resultado no teste (2009):

| Modelo | Parâmetros | Acurácia, solo 66 mm | Acurácia, solos variados | Crítico detectado, solos variados |
|---|---|---|---|---|
| Local (treinado só com 66 mm) | 435 | 85,3% | 83,9% | 83% |
| **Geral (TAW como entrada)** | 451 | 85,0% | 86,3% | 93% |
| Geral, 32 neurônios | 899 | 85,9% | 85,5% | 94% |
| Geral + janelas divididas pelo TAW | 579 | 86,0% | 86,4% | 96% |

- O modelo geral empata com o local no solo dele e ganha nos outros. O maior ganho é deixar escapar menos dias críticos.
- Redes maiores ou features extras rendem no máximo 1 ponto e dobram o tamanho, então não compensam.

**Decisão (18/10/2026): modelo geral, MLP 24-16-3, com o TAW como primeira feature.**

---

## 7. Por que a acurácia caiu de 90,7% para cerca de 85%

Não foi por causa do solo variável: o modelo local, num solo fixo, também cai para 85% quando a poda passa a ser em qualquer data.

O 90,7% da primeira versão era otimista. Todos os ciclos começavam em 15/01 ou 15/07, sempre na mesma época de chuva e de seca, e isso facilitava. Com a poda em qualquer época, o problema fica mais difícil e mais parecido com o campo.

**Para o artigo:** reportar o número do cenário realista (84,6% de acurácia e 88,5% de revocação do crítico na MLP final) e explicar a diferença.

---

## 8. Como ficou

- **PC** (`modelo/`): o `01_gerar_dataset.py` sorteia solo, raiz, poda e produtor para cada ciclo (467 ciclos, cerca de 50.900 dias). O `02` compara e o `03` treina e exporta os pesos.
- **Placa** (`firmware/modelo/`): o `config_local.py` é o único arquivo que muda por local. Ele guarda a água disponível do solo, a profundidade da raiz e a data da poda, e calcula `taw()`, `dap()` e `kc()`.
- **Limite de uso:** o modelo foi treinado com TAW de 25 a 140 mm. Fora disso, a resposta não é confiável.

## 9. E a lâmina de irrigação?

O `classificar(x)` devolve só a **classe** do dia, não quantos mm irrigar. A lâmina pode sair do `Dr` (quanto o solo está seco): pelo FAO-56, `lâmina = Dr ÷ eficiência do gotejamento (~0,9)`. Discutimos três caminhos:

| Caminho | Como | Resultado do teste |
|---|---|---|
| Rede com 2 saídas | A MLP também prevê o Dr em mm | Erro médio de 5,0 mm (90% dos dias até 12,4 mm), com Dr médio de 21 mm. Para achar o crítico, o classificador continua melhor (89% contra 69% a 75%). |
| Balanço FAO-56 na placa | A placa guarda o histórico desde a poda e calcula o Dr exato pela fórmula | Exato, sem rede neural. |
| Só a classe | Lâmina fora do escopo por enquanto | — |

Ponto a discutir: como o rótulo vem de uma fórmula, a placa poderia calcular o Dr exato se tivesse todo o histórico desde a poda. A rede se justifica quando falta parte do histórico (ela usa só 14 dias) ou quando entrarem sensores de umidade reais. Isso muda o que o artigo defende.

**Decisão (19/10/2026): só a classe por enquanto**, até falar com o Prof. Max.

## 10. Pendências

- Confirmar com o Prof. Max as faixas de solo e raiz, a fração p = 0,35 e o limite Ks = 0,5.
- Medir o tempo de inferência na BitDogLab.
- Validar com tensiômetros em campo (Sprint 5). Hoje a acurácia mostra que a rede aprende o balanço do FAO-56, não que acerta o campo.
- Justificar no relatório a troca de C/TFLite para MicroPython.
- Decidir com o Prof. Max se a placa deve indicar a lâmina (seção 9).
