# AgroEdge: dataset e classificador de estado hídrico (uva Itália)

Rodar em ordem (Python 3, `pip install pandas scikit-learn tabulate tensorflow`):

1. `python3 01_gerar_dataset.py` gera `dados/dataset_uva.csv` a partir de `dados/petrolina_clima_2000_2009.csv` (17.440 dias rotulados).
2. `python3 02_comparar_modelos.py` treina MLP, SVM e Random Forest e gera `resultados_comparacao.md`.
3. `python3 03_treinar_mlp_int8.py` treina a MLP no Keras, quantiza em INT8 e gera `../firmware/modelo/*.h` e `resultados_mlp_int8.md`.

## Regra de rotulagem
Balanço hídrico diário do solo do FAO-56 (cap. 8, eq. 84 e 85), sobre o clima real de Petrolina 2000–2009 (DataClimaBR):
`Dr(hoje) = Dr(ontem) − chuva − irrigação + Ks·Kc·ET₀`, limitado entre 0 e TAW.

| Classe | Condição | Significado |
|---|---|---|
| adequado | Dr ≤ RAW | planta sem estresse (Ks = 1) |
| déficit moderado | RAW < Dr ≤ Dr(Ks = 0,5) | estresse leve a médio |
| déficit crítico | Dr > Dr(Ks = 0,5) | evapotranspiração caiu mais da metade |

A irrigação é **simulada**: em cada ciclo, 8 "produtores" com regras diferentes (irrigar por gatilho de umidade ou em turno fixo, com lâminas e falhas variadas). É isso que gera as três classes; com só a chuva, Petrolina daria "crítico" quase o ano todo.

## Suposições (ASSUMIDO, a validar com o Prof. Max)
Ficam no topo do `01_gerar_dataset.py`; troque e rode de novo.
- Podas em 15/01 e 15/07, ciclo de 109 dias (o ciclo vem de Ávila Netto et al., 2000; as datas são suposição). Dias fora do ciclo não entram.
- Raiz efetiva de 0,6 m e água disponível de 0,11 m³/m³ (solo franco-arenoso, FAO-56 tab. 19): TAW = 66 mm.
- Fração p = 0,35 para uva de mesa (FAO-56 tab. 22, conferir): RAW = 23 mm.
- Limite do crítico em Ks = 0,5.
- Os 2 dias faltantes no CSV (08/10/2000 e 14/10/2001) foram interpolados (coluna `interpolado`).

## Features (23, nesta ordem no firmware)
dap, kc, ETo, ETc, chuva, irrigacao, Tmax, Tmin, UR, vento, radiacao, dias_sem_irrigar, e somas móveis de 3, 7 e 14 dias de ETc, chuva e irrigação, mais saldo (chuva + irrigação − ETc) de 7 e 14 dias.
O firmware precisa saber a irrigação aplicada (registro da válvula ou hidrômetro) e guardar 14 dias de histórico.

## Divisão
Por período: treino 2000–2006, validação 2007–2008, teste 2009. Normalização calculada só no treino. Hiperparâmetros escolhidos pela validação (F1 macro).

## Limitações
- O rótulo é sintético: a acurácia mostra que o modelo aprende o balanço do FAO-56, não que acerta o campo. A prova real é com tensiômetros (Sprint 5).
- O clima vem de grade interpolada (BR-DWGD), não dos sensores do protótipo.
- A memória do SVM e da RF é estimativa dos parâmetros em float32, sem o código de inferência.
