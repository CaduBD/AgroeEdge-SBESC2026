# MLP 23-16-3: float × INT8 (teste = 2009)

| modelo | acurácia | F1 macro | rec. adequado | rec. moderado | rec. crítico |
|---|---|---|---|---|---|
| float32 (Keras) | 0.907 | 0.871 | 0.918 | 0.834 | 0.940 |
| INT8 (TFLite) | 0.906 | 0.869 | 0.922 | 0.821 | 0.926 |

Tamanho do .tflite INT8: 2952 bytes. Concordância float × INT8: 0.990.

## Matriz de confusão float32 (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1136 |        102 |         0 |
| moderado |         25 |        242 |        23 |
| critico  |          0 |         13 |       203 |

## Matriz de confusão INT8 (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1142 |         96 |         0 |
| moderado |         31 |        238 |        21 |
| critico  |          0 |         16 |       200 |

