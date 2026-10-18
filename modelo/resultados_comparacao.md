# Comparação MLP × SVM × Random Forest (teste = 2009)

| modelo        | config                  |   acuracia |   f1_macro |   rec_adequado |   rec_moderado |   rec_critico |   memoria_kb |   us_por_amostra_pc |
|:--------------|:------------------------|-----------:|-----------:|---------------:|---------------:|--------------:|-------------:|--------------------:|
| MLP           | (16,)                   |      0.914 |      0.873 |          0.941 |          0.838 |         0.861 |        1.699 |               0.448 |
| SVM (RBF)     | C=10, 3512 vetores      |      0.907 |      0.866 |          0.918 |          0.831 |         0.944 |      342.98  |             195.177 |
| Random Forest | 100 árvores, prof. None |      0.924 |      0.87  |          0.965 |          0.797 |         0.856 |     1000.82  |              27.313 |

## Matriz de confusão: MLP (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1165 |         73 |         0 |
| moderado |         35 |        243 |        12 |
| critico  |          0 |         30 |       186 |

## Matriz de confusão: SVM (RBF) (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1137 |        100 |         1 |
| moderado |         15 |        241 |        34 |
| critico  |          0 |         12 |       204 |

## Matriz de confusão: Random Forest (linhas = real, colunas = predito)

|          |   adequado |   moderado |   critico |
|:---------|-----------:|-----------:|----------:|
| adequado |       1195 |         43 |         0 |
| moderado |         32 |        231 |        27 |
| critico  |          0 |         31 |       185 |

Memória: estimativa em float32 dos parâmetros embarcados, sem o código que faz a inferência.
