"""
AgroEdge - confere no PC o caminho completo do firmware (historico.py -> classificador.py -> lamina.py).

1. Repete os ciclos de 2009 do dataset dia a dia pelo historico.py, como a placa faria,
   e confere se as entradas montadas na placa são as mesmas do dataset de treino.
2. Compara o déficit que a placa calcula com o Dr do rótulo.
3. Fecha o laço: simula a placa irrigando sozinha (classe decide se irriga, lamina.py decide
   quanto) nos mesmos ciclos, e mostra em que estado o solo ficou e quanta água foi usada.

Uso: python3 04_conferir_firmware.py   (depois do 01 e do 03)
"""
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "firmware", "modelo"))
gerar = import_module("01_gerar_dataset")
classificador = import_module("classificador")
historico = import_module("historico")
lamina = import_module("lamina")
cfg = import_module("config_local")

FEATURES = list(classificador.FEATURES)
NOMES = ["adequado", "moderado", "critico"]


def repetir_dataset(te):
    """Parte 1 e 2: o histórico da placa a partir dos mesmos dados de cada dia."""
    x_placa, deficits = [], []
    for _, ciclo in te.groupby("cenario", sort=False):
        h = historico.Historico(arquivo=None)
        for r in ciclo.itertuples():
            x, d = h.fechar_dia(r.TAW, r.dap, r.ETo, r.chuva, r.irrigacao,
                                r.Tmax, r.Tmin, r.UR, r.vento, r.radiacao)
            x_placa.append(x)
            deficits.append(d)
    return np.array(x_placa), np.array(deficits)


def laco_fechado(te, clima):
    """Parte 3: a placa decide a irrigação. O Dr 'real' segue o FAO-56 completo (com Ks)."""
    linhas = []
    for _, ciclo in te.groupby("cenario", sort=False):
        taw = ciclo.TAW.iloc[0]
        _, raw, dr_critico = gerar.solo(taw / 1000, 1.0)        # mesmo TAW, só para RAW e crítico
        dias = clima.loc[ciclo.data.iloc[0]:].iloc[:len(ciclo)]
        h = historico.Historico(arquivo=None)
        dr, irr_amanha = 0.0, 0.0
        for dap, (_, c) in enumerate(dias.iterrows()):
            irr = irr_amanha          # da lâmina bruta aplicada, chega à raiz a líquida
            etc = gerar.kc_uva(dap) * c.ETo
            ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / ((1 - gerar.P_DEPLECAO) * taw))
            dr = min(max(dr - c.Precipitacao - irr + ks * etc, 0.0), taw)
            real = 0 if dr <= raw else (1 if dr <= dr_critico else 2)
            x, deficit = h.fechar_dia(taw, dap, c.ETo, c.Precipitacao, irr, c.Tmax, c.Tmin,
                                      c.Umidade_Relativa, c.Velocidade_Vento, c.Radiacao_Solar)
            classe = classificador.classificar(x)[0]
            irr_amanha = lamina.recomendar(classe, deficit)[0]
            linhas.append(dict(real=real, irrigacao=irr, ETc=etc, chuva=c.Precipitacao))
    return pd.DataFrame(linhas)


def main():
    df = pd.read_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"), parse_dates=["data"])
    te = df[df.split == "teste"]

    x_placa, deficits = repetir_dataset(te)
    erro = np.abs(x_placa - te[FEATURES].values).max(axis=0)
    print("1. Entradas montadas pelo historico.py x dataset (maior diferença por entrada):")
    for nome, e in zip(FEATURES, erro):
        print(f"   {nome:16s} {e:.4f}")
    p = np.array([classificador.classificar(list(x))[0] for x in x_placa])
    print(f"   acurácia com as entradas da placa: {(p == te.classe.values).mean():.3f}\n")

    dif = deficits - te.Dr.values
    print("2. Déficit do lamina.py x Dr do rótulo (mm):")
    print(f"   erro médio {np.abs(dif).mean():.2f} | 90% dos dias até {np.percentile(np.abs(dif), 90):.2f}"
          f" | maior {np.abs(dif).max():.2f} | Dr médio {te.Dr.mean():.1f}\n")

    clima = gerar.carregar_clima()
    r = laco_fechado(te, clima)
    dias = r.real.value_counts(normalize=True).reindex(range(3), fill_value=0)
    print(f"3. Placa irrigando sozinha nos {te.cenario.nunique()} ciclos de 2009:")
    for k in range(3):
        print(f"   {NOMES[k]:9s} {dias[k]:.1%} dos dias")
    print(f"   irrigação líquida {r.irrigacao.sum() / r.ETc.sum():.0%} da ETc do período"
          f" (chuva {r.chuva.sum() / r.ETc.sum():.0%})")
    print(f"   no dataset (produtores simulados): adequado {(te.classe == 0).mean():.1%},"
          f" crítico {(te.classe == 2).mean():.1%}")


if __name__ == "__main__":
    main()
