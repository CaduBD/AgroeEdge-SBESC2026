# MLP 23-16-3: Keras × classificador.py (teste = 2009)

| versão | acurácia | F1 macro | rec. adequado | rec. moderado | rec. crítico |
|---|---|---|---|---|---|
| Keras (float32) | 0.907 | 0.871 | 0.918 | 0.834 | 0.940 |
| classificador.py (MicroPython) | 0.907 | 0.871 | 0.918 | 0.834 | 0.940 |

Concordância Keras × classificador.py: 1.0000.
Parâmetros da rede: 435, mais 46 de normalização; em array('f') ocupam 1924 bytes.

A conferência roda no PC (CPython); na placa o MicroPython calcula em float32, o que pode mudar o último dígito das saídas, mas não deve mudar a classe.

## Matriz de confusão (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1136 |        102 |         0 |
| moderado |         25 |        242 |        23 |
| critico  |          0 |         13 |       203 |
