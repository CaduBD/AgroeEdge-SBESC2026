"""
AgroEdge - comparação MLP × SVM × Random Forest no mesmo dataset (sprint de ML).
Normalização calculada só no treino. Split por período (treino 2000-06, validação 07-08, teste 2009).
Memória = estimativa do que o modelo ocupa embarcado em float32 (ver função memoria_*).

Uso: python3 02_comparar_modelos.py  -> imprime a tabela e salva resultados_comparacao.md
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

AQUI = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["TAW", "dap", "kc", "ETo", "ETc", "chuva", "irrigacao", "Tmax", "Tmin", "UR",
            "vento", "radiacao", "dias_sem_irrigar",
            "ETc_3d", "chuva_3d", "irrigacao_3d", "ETc_7d", "chuva_7d", "irrigacao_7d", "saldo_7d",
            "ETc_14d", "chuva_14d", "irrigacao_14d", "saldo_14d"]
NOMES = ["adequado", "moderado", "critico"]


def memoria_mlp(m):
    return sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_)) * 4


def memoria_svm(m):
    # vetores de suporte (float32) + coeficientes duais + interceptos
    return (m.support_vectors_.size + m.dual_coef_.size + m.intercept_.size) * 4


def memoria_rf(m):
    # por nó: índice da feature (2 B) + limiar (4 B) + 2 filhos (2 × 2 B); folha guarda a classe (1 B)
    total = 0
    for arv in m.estimators_:
        t = arv.tree_
        internos = int((t.children_left != -1).sum())
        folhas = t.node_count - internos
        total += internos * 10 + folhas * 1
    return total


def main():
    df = pd.read_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"))
    tr, va, te = (df[df.split == s] for s in ("treino", "validacao", "teste"))
    sc = StandardScaler().fit(tr[FEATURES])
    Xtr, Xva, Xte = (sc.transform(d[FEATURES]) for d in (tr, va, te))
    ytr, yva, yte = tr.classe.values, va.classe.values, te.classe.values

    # hiperparâmetros escolhidos pela validação (busca pequena, ver candidatos)
    candidatos = {
        "MLP": [MLPClassifier(hidden_layer_sizes=h, max_iter=600, early_stopping=True,
                              random_state=0) for h in [(16,), (32,), (16, 8), (32, 16)]],
        "SVM (RBF)": [SVC(C=c, gamma="scale", class_weight="balanced") for c in (1, 10)],
        "Random Forest": [RandomForestClassifier(n_estimators=n, max_depth=d, class_weight="balanced",
                                                 random_state=0, n_jobs=-1)
                          for n, d in [(20, 8), (50, 10), (100, None)]],
    }
    linhas, matrizes = [], {}
    for nome, lista in candidatos.items():
        melhor = None
        for m in lista:
            # SVM e RF compensam o desbalanceamento com class_weight="balanced";
            # o MLPClassifier não tem essa opção (na MLP final, no Keras, ela é usada)
            m.fit(Xtr, ytr)
            f1 = f1_score(yva, m.predict(Xva), average="macro")
            if melhor is None or f1 > melhor[0]:
                melhor = (f1, m)
        m = melhor[1]
        t0 = time.perf_counter()
        p = m.predict(Xte)
        dt = (time.perf_counter() - t0) / len(Xte) * 1e6
        mem = {"MLP": memoria_mlp, "SVM (RBF)": memoria_svm, "Random Forest": memoria_rf}[nome](m)
        rec = recall_score(yte, p, average=None)
        cfg = (m.hidden_layer_sizes if nome == "MLP" else
               f"C={m.C}, {len(m.support_vectors_)} vetores" if nome.startswith("SVM") else
               f"{m.n_estimators} árvores, prof. {m.max_depth}")
        linhas.append(dict(modelo=nome, config=str(cfg), acuracia=accuracy_score(yte, p),
                           f1_macro=f1_score(yte, p, average="macro"),
                           rec_adequado=rec[0], rec_moderado=rec[1], rec_critico=rec[2],
                           memoria_kb=mem / 1024, us_por_amostra_pc=dt))
        matrizes[nome] = confusion_matrix(yte, p)

    res = pd.DataFrame(linhas)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    with open(os.path.join(AQUI, "resultados_comparacao.md"), "w") as f:
        f.write("# Comparação MLP × SVM × Random Forest (teste = 2009)\n\n")
        f.write(res.round(3).to_markdown(index=False) + "\n\n")
        for nome, cm in matrizes.items():
            f.write(f"## Matriz de confusão: {nome} (linhas = real, colunas = predito)\n\n")
            f.write(pd.DataFrame(cm, index=NOMES, columns=NOMES).to_markdown() + "\n\n")
        f.write("Memória: estimativa em float32 dos parâmetros embarcados, sem o código que faz a inferência.\n")
    for nome, cm in matrizes.items():
        print(nome); print(cm)


if __name__ == "__main__":
    main()
