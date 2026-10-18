"""
AgroEdge - treino da MLP (Keras), quantização INT8 e exportação para o RP2040.
Mesmo dataset, features e split do 02_comparar_modelos.py.

Gera nesta pasta:
  agroedge_mlp_int8.tflite        modelo quantizado (entrada e saída int8)
  firmware/modelo/agroedge_mlp_int8.h    modelo como array C (no padrão do cnn_mnist_tinyML)
  firmware/modelo/agroedge_normalizacao.h  médias e desvios do treino (padrão do tiny_ml_iris)
  resultados_mlp_int8.md          float × INT8: acurácia, matriz de confusão, tamanho
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
FEATURES = import_module("02_comparar_modelos").FEATURES
NOMES = ["adequado", "moderado", "critico"]
AQUI = os.path.dirname(os.path.abspath(__file__))
OCULTA = 16          # 1 camada de 16 neurônios (melhor F1 na validação do script 02)


def main():
    tf.keras.utils.set_random_seed(0)
    df = pd.read_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"))
    tr, va, te = (df[df.split == s] for s in ("treino", "validacao", "teste"))
    media = tr[FEATURES].mean().values.astype(np.float32)
    desvio = tr[FEATURES].std(ddof=0).values.astype(np.float32)
    norm = lambda d: ((d[FEATURES].values - media) / desvio).astype(np.float32)
    Xtr, Xva, Xte = norm(tr), norm(va), norm(te)
    ytr, yva, yte = tr.classe.values, va.classe.values, te.classe.values
    pesos = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr)))

    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(OCULTA, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200, batch_size=64,
               class_weight=pesos, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    p_float = modelo.predict(Xte, verbose=0).argmax(1)

    # quantização INT8 completa, com dados de treino como amostra representativa
    def representativo():
        for i in np.random.default_rng(0).choice(len(Xtr), 500, replace=False):
            yield [Xtr[i:i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    open(os.path.join(AQUI, "agroedge_mlp_int8.tflite"), "wb").write(tfl)

    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    ent, sai = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = ent["quantization"]
    p_int8 = []
    for x in Xte:
        q = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
        it.set_tensor(ent["index"], q[None, :])
        it.invoke()
        p_int8.append(it.get_tensor(sai["index"])[0].argmax())
    p_int8 = np.array(p_int8)

    # exportação para C
    fw = os.path.join(AQUI, "..", "firmware", "modelo")
    os.makedirs(fw, exist_ok=True)
    corpo = ",".join(("\n  " if i % 12 == 0 else " ") + f"0x{b:02x}" for i, b in enumerate(tfl))
    with open(os.path.join(fw, "agroedge_mlp_int8.h"), "w") as f:
        f.write("// Gerado por modelo/03_treinar_mlp_int8.py - não editar à mão\n#pragma once\n\n")
        f.write(f"alignas(16) const unsigned char agroedge_mlp_int8[] = {{{corpo}\n}};\n")
        f.write(f"const unsigned int agroedge_mlp_int8_len = {len(tfl)};\n")
    with open(os.path.join(fw, "agroedge_normalizacao.h"), "w") as f:
        f.write("// Gerado por modelo/03_treinar_mlp_int8.py - médias e desvios do conjunto de TREINO\n")
        f.write(f"#pragma once\n#define AGROEDGE_N_FEATURES {len(FEATURES)}\n")
        f.write("// ordem: " + ", ".join(FEATURES) + "\n")
        f.write("static const float agroedge_media[AGROEDGE_N_FEATURES] = {"
                + ", ".join(f"{v:.6f}f" for v in media) + "};\n")
        f.write("static const float agroedge_desvio[AGROEDGE_N_FEATURES] = {"
                + ", ".join(f"{v:.6f}f" for v in desvio) + "};\n")
        f.write(f"static const float agroedge_in_scale = {s_in:.8f}f;\nstatic const int agroedge_in_zero_point = {z_in};\n")

    linhas = []
    with open(os.path.join(AQUI, "resultados_mlp_int8.md"), "w") as f:
        f.write(f"# MLP {len(FEATURES)}-{OCULTA}-3: float × INT8 (teste = 2009)\n\n")
        for nome, p in (("float32 (Keras)", p_float), ("INT8 (TFLite)", p_int8)):
            rec = recall_score(yte, p, average=None)
            linha = (f"| {nome} | {accuracy_score(yte, p):.3f} | {f1_score(yte, p, average='macro'):.3f} | "
                     f"{rec[0]:.3f} | {rec[1]:.3f} | {rec[2]:.3f} |")
            linhas.append(linha)
        f.write("| modelo | acurácia | F1 macro | rec. adequado | rec. moderado | rec. crítico |\n|---|---|---|---|---|---|\n")
        f.write("\n".join(linhas) + "\n\n")
        f.write(f"Tamanho do .tflite INT8: {len(tfl)} bytes. Concordância float × INT8: "
                f"{(p_float == p_int8).mean():.3f}.\n\n")
        for nome, p in (("float32", p_float), ("INT8", p_int8)):
            f.write(f"## Matriz de confusão {nome} (linhas = real, colunas = predito)\n\n")
            f.write(pd.DataFrame(confusion_matrix(yte, p), index=NOMES, columns=NOMES).to_markdown() + "\n\n")
    print(open(os.path.join(AQUI, "resultados_mlp_int8.md")).read())


if __name__ == "__main__":
    main()
