# AgroEdge - teste rápido do classificador na BitDogLab.
# Copie classificador.py, agroedge_pesos.py, config_local.py e este arquivo para a placa e rode este.
# Mostra a classe de 6 dias do conjunto de teste (2009, solos de 35 a 111 mm de TAW),
# o tempo de cada inferência e a memória livre.

import gc
import time
import classificador
import config_local

# no PC (Python normal) não existem gc.mem_free nem time.ticks_us; aí usamos equivalentes
NA_PLACA = hasattr(gc, "mem_free")
if NA_PLACA:
    agora_us, diferenca_us = time.ticks_us, time.ticks_diff
else:
    agora_us = lambda: time.perf_counter_ns() // 1000
    diferenca_us = lambda fim, inicio: fim - inicio

# (features na ordem de classificador.FEATURES, classe esperada)
DIAS = (
    ((42.71, 55, 0.7355, 5.282, 3.885, 0, 0, 29.45, 20.38, 56.59, 3.189, 18.58, 1, 12.42, 0, 20.12, 27.78, 0.405, 20.12, -7.249, 51.87, 2.042, 49.69, -0.1408), 0),  # 2009-08-13, TAW 43 mm
    ((111.4, 103, 0.556, 4.968, 2.762, 6.46, 0, 33.57, 21.78, 67.79, 1.941, 19.56, 4, 6.398, 16.3, 0, 21.27, 17.57, 15.25, 11.55, 52.54, 17.57, 45, 10.04), 0),  # 2009-10-13, TAW 111 mm
    ((40.58, 59, 0.7346, 3.905, 2.869, 0.912, 0, 30.35, 22.67, 73.42, 1.777, 16.81, 9, 8.024, 0.952, 0, 18.51, 7.803, 0, -10.71, 39.44, 13.04, 6.231, -20.17), 1),  # 2009-04-29, TAW 41 mm
    ((100, 69, 0.7212, 6.636, 4.786, 0, 13.43, 35.15, 23, 44.15, 1.76, 26.21, 0, 14.94, 2.474, 13.43, 34.25, 4.668, 24.3, -5.281, 61.16, 22.24, 30, -8.914), 1),  # 2009-10-20, TAW 100 mm
    ((34.58, 51, 0.7338, 7.078, 5.194, 0, 0, 36.1, 24.16, 43.41, 2.362, 24.62, 1, 15.76, 0, 7.461, 35.02, 0, 13.95, -21.07, 68.63, 0, 30.71, -37.92), 2),  # 2009-09-21, TAW 35 mm
    ((99.53, 90, 0.641, 5.393, 3.457, 0, 1.89, 30.89, 21.35, 56.4, 3.109, 19.34, 0, 9.904, 0, 3.637, 22.87, 0.056, 5.573, -17.24, 44.03, 0.254, 12.27, -31.51), 2),  # 2009-07-15, TAW 100 mm
)

gc.collect()
print("memoria livre:", gc.mem_free() if NA_PLACA else "(só na placa)", "bytes" if NA_PLACA else "")
acertos = 0
for x, esperada in DIAS:
    t0 = agora_us()
    classe, saidas = classificador.classificar(x)
    dt = diferenca_us(agora_us(), t0)
    acertos += classe == esperada
    print("%-17s esperado %-17s %6d us" % (classificador.CLASSES[classe],
                                          classificador.CLASSES[esperada], dt))
print("acertos: %d de %d" % (acertos, len(DIAS)))
print("TAW deste local (config_local.py): %.1f mm" % config_local.taw())
