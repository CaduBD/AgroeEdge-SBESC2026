# Guia passo a passo: como o modelo do AgroEdge funciona

Este guia explica, na ordem em que as coisas acontecem, tudo o que está em `modelo/` e em `firmware/modelo/`.

## 0. A ideia geral

O fluxo inteiro é uma linha de montagem:

```
clima real de Petrolina (CSV)
   → 01: simula irrigação e calcula a água no solo dia a dia → rótulo (adequado / moderado / crítico)
   → 02: treina MLP, SVM e Random Forest e compara
   → 03: treina a MLP final e exporta os pesos para MicroPython
   → firmware/modelo/: classificador.py + agroedge_pesos.py, o que a BitDogLab usa para classificar
```

A pergunta que o classificador responde é: **"com o clima e a irrigação dos últimos dias, como está a água no solo hoje?"**

---

## 1. `01_gerar_dataset.py`: criar os dados rotulados

### 1.1 Por que precisamos simular
Para treinar um classificador, cada dia precisa de uma resposta certa (o rótulo). Não temos medição de umidade do solo de 2000 a 2009, então calculamos essa resposta com o **balanço hídrico do FAO-56**, que é o método padrão da agronomia. E como numa cultura irrigada quem define o estado do solo é a irrigação, simulamos produtores irrigando de jeitos diferentes.

### 1.2 Os parâmetros (topo do arquivo)
| Parâmetro | Valor | O que é |
|---|---|---|
| `PODAS` | 15/01 e 15/07 | início de cada ciclo da uva (ASSUMIDO) |
| `CICLO_DIAS` | 109 | duração do ciclo, de Ávila Netto et al. (2000) |
| `ZR_M` | 0,6 m | profundidade das raízes (ASSUMIDO) |
| `AGUA_DISP_M3M3` | 0,11 | quanta água o solo segura por metro (ASSUMIDO) |
| `P_DEPLECAO` | 0,35 | fração que a planta tira sem sofrer (FAO-56, ASSUMIDO) |
| `KS_CRITICO` | 0,5 | a partir de onde chamamos de crítico (ASSUMIDO) |
| `CENARIOS_POR_CICLO` | 8 | quantos produtores simulados por ciclo |
| `SEMENTE` | 42 | faz o sorteio sair sempre igual (resultado reproduzível) |

Com eles, o script calcula três números que definem o "tanque de água" do solo:

- **TAW** (água total disponível) = 1000 × 0,11 × 0,6 = **66 mm**. É o tanque cheio.
- **RAW** (água facilmente disponível) = 0,35 × 66 = **23,1 mm**. Até gastar isso, a planta não sofre.
- **Dr crítico** = 66 − 0,5 × (1 − 0,35) × 66 = **44,5 mm**. Passando disso, a planta transpira menos da metade do normal.

Pense no solo como uma caixa d'água de 66 mm. `Dr` (depleção) é **quanto já foi gasto**. Dr = 0 é caixa cheia; Dr = 66 é caixa vazia.

### 1.3 `kc_uva(dap)`
A equação de Ávila Netto: `Kc = −0,00008·DAP² + 0,0089·DAP + 0,488`. DAP é o dia depois da poda (0 a 108). O Kc começa em 0,49, sobe até ~0,74 e cai no fim do ciclo. Multiplicando por ET₀, temos quanto a uva consome: **ETc = Kc × ET₀**.

### 1.4 `carregar_clima()`
Lê o CSV de Petrolina e preenche os 2 dias que faltam (08/10/2000 e 14/10/2001) pela média entre o dia anterior e o seguinte. A coluna `interpolado` marca esses dias.

### 1.5 `politica_aleatoria(rng)`: o produtor simulado
Sorteia um de dois tipos de produtor:
- **Gatilho:** irriga quando a caixa passou de um certo nível de gasto (de 0,3 a 2,4 × RAW) e repõe uma parte do que faltava (30% a 110%).
- **Turno fixo:** irriga a cada N dias (1 a 7) uma lâmina proporcional ao ETc.

Os dois têm uma chance de "esquecer" de irrigar no dia (0 a 50%). Isso cria produtores bons e ruins, e é o que faz aparecerem as três classes.

### 1.6 `simular_ciclo(...)`: o coração do script
Para cada dia do ciclo:

1. Calcula `Kc` e `ETc`.
2. O produtor decide se irriga, olhando o estado da manhã.
3. Calcula o **Ks** (coeficiente de estresse): se `Dr ≤ RAW`, Ks = 1 (sem estresse). Se passou, Ks cai em linha reta até 0 quando o solo esvazia. A planta com sede fecha os estômatos e consome menos.
4. Atualiza a caixa (equação 85 do FAO-56):
   `Dr_hoje = Dr_ontem − chuva − irrigação + Ks × ETc`
5. Limita `Dr` entre 0 e 66. Água a mais drena para baixo e se perde.
6. Rotula o dia:
   - `Dr ≤ 23,1` → **0 = adequado**
   - `23,1 < Dr ≤ 44,5` → **1 = déficit moderado**
   - `Dr > 44,5` → **2 = déficit crítico**

**Exemplo de um dia:** ontem Dr = 20 mm. Hoje ETo = 6 mm, Kc = 0,7, então ETc = 4,2 mm. Não choveu e não irrigou. Como 20 ≤ 23,1, Ks = 1. Dr = 20 − 0 − 0 + 4,2 = **24,2 mm** → passou de 23,1, então hoje é "déficit moderado".

### 1.7 `adicionar_janelas(df)`: a memória dos últimos dias
A planta não sabe só do dia de hoje: o estado do solo é a soma do passado. Então, para cada dia, somamos ETc, chuva e irrigação dos últimos **3, 7 e 14 dias**, e o **saldo** (`chuva + irrigação − ETc`) de 7 e 14 dias. São essas somas que o firmware vai precisar guardar.

### 1.8 `main()`
Roda 10 anos × 2 podas × 8 produtores = **160 ciclos**, cerca de 17.440 dias. Separa por ano:
- **treino:** 2000 a 2006 (o modelo aprende aqui)
- **validação:** 2007 e 2008 (escolhemos a configuração aqui)
- **teste:** 2009 (nota final, ano que o modelo nunca viu)

Separar por ano, e não sorteando dias, evita "cola": dias vizinhos são muito parecidos, e se caíssem em grupos diferentes a nota sairia falsamente alta.

Saída: `dados/dataset_uva.csv`.

---

## 2. `02_comparar_modelos.py`: qual algoritmo usar

### 2.1 As 23 features (entradas)
`FEATURES` é a lista do que o modelo recebe: dia do ciclo (`dap`), `kc`, `ETo`, `ETc`, chuva e irrigação do dia, Tmax, Tmin, UR, vento, radiação, `dias_sem_irrigar` e as somas de 3/7/14 dias. **Note que `Dr` não está na lista**: o modelo tem que adivinhar o estado do solo só com o que o sensor e o histórico mostram.

### 2.2 Normalização
`StandardScaler` transforma cada feature em `(x − média) / desvio`. Assim, chuva em mm e UR em % ficam na mesma escala, e a rede aprende melhor. A média e o desvio são calculados **só no treino**, porque o modelo não pode "espiar" os dados de teste.

### 2.3 Os três candidatos
- **MLP** (rede neural): testa 4 tamanhos de camadas escondidas.
- **SVM com kernel RBF:** testa C = 1 e 10.
- **Random Forest:** testa 3 combinações de número de árvores e profundidade.

Cada um é treinado no treino e avaliado na validação pelo **F1 macro**, a média do F1 das três classes. Usamos F1 macro, e não acurácia, porque a acurácia pode enganar quando uma classe é bem maior que as outras.

### 2.4 As métricas no teste (2009)
- **Acurácia:** % de dias certos.
- **Revocação por classe:** dos dias que eram realmente "crítico", quantos o modelo achou. É a mais importante, porque errar um crítico é deixar a planta sofrer.
- **Matriz de confusão:** linha = verdade, coluna = resposta do modelo. A diagonal são os acertos.
- **Memória estimada:**
  - MLP: nº de pesos × 4 bytes.
  - SVM: precisa guardar milhares de "vetores de suporte" (dias de treino) inteiros.
  - RF: cada nó de cada árvore ocupa ~10 bytes.

Resultado: os três acertam ~91%, mas o SVM ocupa 343 KB e a RF 1 MB. A MLP ocupa 1,7 KB. Por isso a escolha é a MLP.

---

## 3. `03_treinar_mlp.py`: a MLP final para a placa

### 3.1 A rede
```
23 entradas → 16 neurônios (ReLU) → 3 saídas (softmax)
```
- Cada neurônio faz `soma(peso × entrada) + bias` e passa pela **ReLU** (`max(0, x)`).
- O **softmax** transforma as 3 saídas em probabilidades que somam 1. A classe escolhida é a de maior valor.
- Total: 23×16 + 16 + 16×3 + 3 = **435 parâmetros**.

### 3.2 O treino
- `class_weight`: erros nas classes raras (crítico é só 13%) pesam mais, para o modelo não as ignorar.
- `EarlyStopping`: para quando a validação para de melhorar por 15 épocas e volta para a melhor versão. Evita decorar o treino (overfitting).

### 3.3 Exportação para MicroPython
Depois do treino, o script tira de dentro do Keras os números da rede (`get_weights()`) e escreve o arquivo `firmware/modelo/agroedge_pesos.py`. Ele contém:
- as médias e os desvios do treino, para normalizar;
- `W1` e `B1`, os pesos e bias da camada escondida;
- `W2` e `B2`, os da camada de saída.

Todos ficam guardados como `array("f", [...])`, vetores de floats de 32 bits, que ocupam 4 bytes por número (cerca de 1,9 KB no total).

**Por que não precisamos de TensorFlow Lite nem de INT8:** a rede é tão pequena que dá para escrever a conta dela à mão em poucas linhas de Python (próxima seção). A conta tem cerca de 420 multiplicações e roda uma vez por dia; mesmo sem FPU, isso é rápido.

### 3.4 A conferência
O script importa o `classificador.py` de verdade, roda os 1.744 dias de 2009 por ele e compara com o Keras: **as duas respostas são iguais em 100% dos dias, com 90,7% de acurácia**. Também rodei num MicroPython compilado com float de 32 bits, como o do RP2040, e o resultado foi o mesmo.

---

## 4. `firmware/modelo/`: o que vai para a BitDogLab

### 4.1 `agroedge_pesos.py` (gerado, não editar)
Só dados: `N_FEATURES`, `N_OCULTA`, `N_CLASSES`, os nomes das features (`FEATURES`, na ordem exata em que o firmware deve montar a entrada) e das classes (`CLASSES`), e os vetores `MEDIA`, `DESVIO`, `W1`, `B1`, `W2`, `B2`.

### 4.2 `classificador.py` (escrito à mão)
Tem uma função, `classificar(x)`, que faz quatro passos:
1. **Normaliza:** `z = (x − média) / desvio` para cada uma das 23 entradas.
2. **Camada escondida:** para cada um dos 16 neurônios, soma `z[i] × W1` + bias e aplica a ReLU.
3. **Camada de saída:** para cada uma das 3 classes, soma `h[j] × W2` + bias.
4. **Escolhe:** devolve o índice da maior saída (0 = adequado, 1 = moderado, 2 = crítico) e as 3 saídas.

O softmax não é calculado na placa, porque ele não muda qual saída é a maior. Isso economiza 3 exponenciais por dia.

### 4.3 `teste_classificador.py`
Roda 6 dias reais de 2009 (2 de cada classe) e mostra a classe prevista, a esperada, o tempo de cada inferência e a memória livre. Serve para confirmar na placa que tudo foi copiado certo e para medir o tempo real.

### 4.4 Como usar no firmware (uma vez por dia)
```python
import classificador

x = montar_features()        # você escreve: sensores + histórico de 14 dias, na ordem de classificador.FEATURES
classe, saidas = classificador.classificar(x)
print(classificador.CLASSES[classe])
```

---

## 5. Como mudar algo e refazer

1. Ajuste um parâmetro no topo do `01_gerar_dataset.py`, por exemplo as datas de poda que o Prof. Max indicar.
2. Rode, na pasta `modelo/`:
   ```
   python 01_gerar_dataset.py
   python 02_comparar_modelos.py
   python 03_treinar_mlp.py
   ```
3. O `agroedge_pesos.py` é gerado de novo. Copie ele para a placa: o `classificador.py` não muda.

Dependências (só no PC, não na placa): `pip install pandas scikit-learn tabulate tensorflow`.
