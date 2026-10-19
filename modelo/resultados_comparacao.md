# Comparação MLP × SVM × Random Forest (teste = 2009)

| modelo        | config                  |   acuracia |   f1_macro |   rec_adequado |   rec_moderado |   rec_critico |   memoria_kb |   us_por_amostra_pc |
|:--------------|:------------------------|-----------:|-----------:|---------------:|---------------:|--------------:|-------------:|--------------------:|
| MLP           | (32, 16)                |      0.91  |      0.888 |          0.945 |          0.845 |         0.851 |        7.262 |               0.387 |
| SVM (RBF)     | C=10, 7631 vetores      |      0.911 |      0.894 |          0.942 |          0.834 |         0.932 |     1222.16  |             459.52  |
| Random Forest | 100 árvores, prof. None |      0.931 |      0.906 |          0.965 |          0.882 |         0.832 |     1969.37  |              13.373 |

## Matriz de confusão: MLP (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2307 |        133 |         0 |
| moderado |        138 |        890 |        25 |
| critico  |          0 |         48 |       274 |

## Matriz de confusão: SVM (RBF) (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2298 |        142 |         0 |
| moderado |        129 |        878 |        46 |
| critico  |          0 |         22 |       300 |

## Matriz de confusão: Random Forest (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       2354 |         86 |         0 |
| moderado |        107 |        929 |        17 |
| critico  |          0 |         54 |       268 |

Memória: estimativa em float32 dos parâmetros embarcados, sem o código que faz a inferência.
