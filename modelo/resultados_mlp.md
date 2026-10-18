# MLP 24-16-3: Keras × classificador.py (teste = 2009)

| versão | acurácia | F1 macro | rec. adequado | rec. moderado | rec. crítico |
|---|---|---|---|---|---|
| Keras (float32) | 0.846 | 0.800 | 0.892 | 0.726 | 0.885 |
| classificador.py (MicroPython) | 0.846 | 0.800 | 0.892 | 0.726 | 0.885 |

Concordância Keras × classificador.py: 1.0000.
Parâmetros da rede: 451, mais 48 de normalização; em array('f') ocupam 1996 bytes.

A conferência roda no PC (CPython); na placa o MicroPython calcula em float32, o que pode mudar o último dígito das saídas, mas não deve mudar a classe.

## Matriz de confusão (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2177 |        261 |         2 |
| moderado |        151 |        765 |       137 |
| critico  |          0 |         37 |       285 |
