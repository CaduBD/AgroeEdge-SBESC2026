"""
AgroEdge - geração do dataset rotulado (uva Itália, Petrolina 2000-2009).

Rótulo = estado hídrico do solo pelo balanço hídrico diário do FAO-56 (cap. 8, eq. 85),
com irrigação SIMULADA (vários "produtores" com regras diferentes), sobre o clima real
do DataClimaBR. Tudo que depende do Prof. Max está marcado como ASSUMIDO.

Uso: python3 01_gerar_dataset.py  -> gera dataset_uva.csv nesta pasta.
"""
import os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV_CLIMA = os.path.join(AQUI, "dados", "petrolina_clima_2000_2009.csv")

# ---------------------------------------------------------------- parâmetros
PODAS = [(1, 15), (7, 15)]   # ASSUMIDO: 2 podas/ano, 15/jan e 15/jul (mês, dia)
CICLO_DIAS = 109             # Ávila Netto et al. (2000), Petrolina
ZR_M = 0.6                   # ASSUMIDO: profundidade efetiva das raízes (m), gotejamento
AGUA_DISP_M3M3 = 0.11        # ASSUMIDO: θCC − θPMP, solo franco-arenoso (FAO-56, tab. 19, valor médio)
P_DEPLECAO = 0.35            # ASSUMIDO: fração p para uva de mesa (FAO-56, tab. 22) - conferir
KS_CRITICO = 0.5             # ASSUMIDO: Ks < 0,5 -> déficit crítico
CENARIOS_POR_CICLO = 8       # nº de produtores simulados por ciclo
SEMENTE = 42

TAW = 1000 * AGUA_DISP_M3M3 * ZR_M          # água total disponível (mm)
RAW = P_DEPLECAO * TAW                      # água facilmente disponível (mm)
DR_CRITICO = TAW - KS_CRITICO * (1 - P_DEPLECAO) * TAW   # Dr onde Ks = KS_CRITICO

CLASSES = {0: "adequado", 1: "deficit_moderado", 2: "deficit_critico"}


def kc_uva(dap):
    """Ávila Netto et al. (2000): Kc(DAP), r² = 0,97."""
    return -8e-5 * dap**2 + 0.0089 * dap + 0.488


def carregar_clima():
    df = pd.read_csv(CSV_CLIMA, parse_dates=["data"]).set_index("data").sort_index()
    completo = pd.date_range(df.index.min(), df.index.max(), freq="D")
    faltando = completo.difference(df.index)
    df = df.reindex(completo).interpolate()       # 2 dias faltantes (ver notas)
    df["interpolado"] = df.index.isin(faltando).astype(int)
    return df


def politica_aleatoria(rng):
    """Um 'produtor' simulado. Mistura bons e maus manejos para cobrir as 3 classes."""
    tipo = rng.choice(["gatilho", "turno_fixo"])
    if tipo == "gatilho":
        return dict(tipo=tipo,
                    gatilho=rng.uniform(0.3, 2.4) * RAW,     # irriga quando Dr passa disso
                    fracao=rng.uniform(0.3, 1.1),             # repõe essa fração de Dr
                    falha=rng.uniform(0.0, 0.5))             # chance de não irrigar no dia
    return dict(tipo=tipo,
                turno=int(rng.integers(1, 8)),                # irriga a cada N dias
                lamina_fator=rng.uniform(0.2, 1.2),           # lâmina = fator × ETc média × N
                falha=rng.uniform(0.0, 0.5))


def simular_ciclo(clima, inicio, pol, rng, cenario_id):
    dias = clima.loc[inicio: inicio + pd.Timedelta(days=CICLO_DIAS - 1)]
    dr = 0.0                       # solo na capacidade de campo na poda (irrigação de pré-poda)
    ultimo = 0
    linhas = []
    for dap, (data, c) in enumerate(dias.iterrows()):
        kc = kc_uva(dap)
        etc = kc * c.ETo
        # decisão de irrigar (com base no estado do início do dia)
        irr = 0.0
        if rng.random() >= pol["falha"]:
            if pol["tipo"] == "gatilho" and dr > pol["gatilho"]:
                irr = pol["fracao"] * dr
            elif pol["tipo"] == "turno_fixo" and dap % pol["turno"] == 0:
                irr = pol["lamina_fator"] * etc * pol["turno"]
        # coeficiente de estresse e balanço (FAO-56 eq. 84 e 85)
        ks = 1.0 if dr <= RAW else max(0.0, (TAW - dr) / ((1 - P_DEPLECAO) * TAW))
        etc_aj = ks * etc
        dr = dr - c.Precipitacao - irr + etc_aj
        dr = min(max(dr, 0.0), TAW)   # excesso drena (DP); não passa do ponto de murcha
        if irr > 0:
            ultimo = dap
        if dr <= RAW:
            classe = 0
        elif dr <= DR_CRITICO:
            classe = 1
        else:
            classe = 2
        linhas.append(dict(data=data, cenario=cenario_id, dap=dap, kc=kc,
                           ETo=c.ETo, ETc=etc, chuva=c.Precipitacao, irrigacao=irr,
                           Tmax=c.Tmax, Tmin=c.Tmin, UR=c.Umidade_Relativa,
                           vento=c.Velocidade_Vento, radiacao=c.Radiacao_Solar,
                           dias_sem_irrigar=dap - ultimo, interpolado=c.interpolado,
                           Dr=dr, classe=classe))
    return linhas


def adicionar_janelas(df):
    """Acumulados que o firmware consegue manter (só passado, por cenário/ciclo)."""
    g = df.groupby(["cenario"], sort=False)
    for n in (3, 7, 14):
        for col in ("ETc", "chuva", "irrigacao"):
            df[f"{col}_{n}d"] = g[col].transform(lambda s: s.rolling(n, min_periods=1).sum())
    for n in (7, 14):
        df[f"saldo_{n}d"] = df[f"chuva_{n}d"] + df[f"irrigacao_{n}d"] - df[f"ETc_{n}d"]
    return df


def main():
    rng = np.random.default_rng(SEMENTE)
    clima = carregar_clima()
    linhas = []
    cid = 0
    for ano in sorted(set(clima.index.year)):
        for mes, dia in PODAS:
            inicio = pd.Timestamp(ano, mes, dia)
            if inicio + pd.Timedelta(days=CICLO_DIAS - 1) > clima.index.max():
                continue
            for _ in range(CENARIOS_POR_CICLO):
                linhas += simular_ciclo(clima, inicio, politica_aleatoria(rng), rng, cid)
                cid += 1
    df = adicionar_janelas(pd.DataFrame(linhas))
    df["ano"] = df["data"].dt.year
    df["split"] = np.where(df.ano <= 2006, "treino", np.where(df.ano <= 2008, "validacao", "teste"))
    df.to_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"), index=False, float_format="%.4f")

    print(f"TAW={TAW:.1f} mm  RAW={RAW:.1f} mm  Dr crítico>{DR_CRITICO:.1f} mm")
    print(f"{len(df)} linhas, {cid} cenários (ciclo × produtor)")
    tab = pd.crosstab(df.split, df.classe.map(CLASSES), normalize="index").round(3)
    print(tab)
    print(df.classe.map(CLASSES).value_counts())


if __name__ == "__main__":
    main()
