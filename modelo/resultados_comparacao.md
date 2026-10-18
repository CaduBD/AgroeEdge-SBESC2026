# Comparação MLP × SVM × Random Forest (teste = 2009)

| modelo        | config                  |   acuracia |   f1_macro |   rec_adequado |   rec_moderado |   rec_critico |   memoria_kb |   us_por_amostra_pc |
|:--------------|:------------------------|-----------:|-----------:|---------------:|---------------:|--------------:|-------------:|--------------------:|
| MLP           | (32, 16)                |      0.858 |      0.812 |          0.916 |          0.737 |         0.817 |        5.387 |               0.469 |
| SVM (RBF)     | C=10, 12072 vetores     |      0.838 |      0.792 |          0.887 |          0.69  |         0.947 |     1226.07  |             711.544 |
| Random Forest | 100 árvores, prof. None |      0.853 |      0.807 |          0.904 |          0.754 |         0.795 |     3546.44  |              12.056 |

## Matriz de confusão: MLP (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2234 |        205 |         1 |
| moderado |        184 |        776 |        93 |
| critico  |          0 |         59 |       263 |

## Matriz de confusão: SVM (RBF) (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2164 |        273 |         3 |
| moderado |        159 |        727 |       167 |
| critico  |          0 |         17 |       305 |

## Matriz de confusão: Random Forest (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2206 |        234 |         0 |
| moderado |        169 |        794 |        90 |
| critico  |          0 |         66 |       256 |

Memória: estimativa em float32 dos parâmetros embarcados, sem o código que faz a inferência.
