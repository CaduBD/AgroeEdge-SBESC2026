# AgroEdge - teste rápido do classificador na BitDogLab.
# Copie classificador.py, agroedge_pesos.py, config_local.py e este arquivo para a placa e rode este.
# Mostra a classe de 6 dias do conjunto de teste (2009, solos de 42 a 100 mm de TAW),
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
    ((41.91, 67, 0.7252, 4.968, 3.603, 6.46, 0, 33.57, 21.78, 67.79, 1.941, 19.56, 2, 8.272, 16.3, 13.53, 26.75, 17.57, 26.07, 16.89, 63.57, 17.57, 52.93, 6.932, 0.6382, 0.4193, 0.4031, 1.517, 0.4193, 0.1654, 2.422, 0.4193, 0.1774, 3.249, 0.4193, -0.2003, 6.819, 0.4289, -1.074), 0),  # 2009-10-13, TAW 42 mm, DAP 67
    ((93.63, 45, 0.7265, 3.461, 2.514, 0.079, 0, 28.17, 20.89, 70.23, 3.133, 10.78, 4, 7.832, 0.148, 0, 20.1, 0.405, 13.83, -5.864, 38.77, 8.716, 27.27, -2.786, 0.2147, 0.0043, -0.0626, 0.414, 0.0931, -0.0298, 0.6009, 0.1599, -0.1499, 0.7716, 0.406, -0.0744, 1.155, 0.76, -0.1039), 0),  # 2009-06-10, TAW 94 mm, DAP 45
    ((42.71, 45, 0.7265, 5.058, 3.675, 0.034, 0, 30.51, 20.88, 60.26, 3.119, 17.74, 3, 11.1, 1.273, 0, 25.7, 1.43, 12.94, -11.33, 49.29, 1.607, 29.28, -18.4, 0.6018, 0.0335, -0.2653, 1.154, 0.0376, -0.4309, 1.671, 0.0482, -0.595, 2.17, 0.0504, -0.8242, 3.156, 0.1133, -0.9875), 1),  # 2009-08-03, TAW 43 mm, DAP 45
    ((100, 35, 0.7015, 6.318, 4.432, 0, 0, 34.01, 22.92, 51.03, 2.516, 22.39, 5, 13.26, 0, 0, 31.77, 0, 13.1, -18.67, 61.92, 0, 37.46, -24.46, 0.3177, 0, -0.1867, 0.6192, 0, -0.2446, 0.8831, 0, -0.3013, 1.124, 0, -0.4394, 1.345, 0, -0.5025), 1),  # 2009-09-16, TAW 100 mm, DAP 35
    ((43.88, 93, 0.6238, 5.309, 3.312, 0, 0, 29.49, 19.92, 59.01, 3.131, 20.28, 33, 10.27, 0, 0, 25.01, 0.364, 0, -24.65, 48.08, 0.874, 0, -47.21, 0.57, 0.0083, -0.5617, 1.096, 0.0199, -1.076, 1.668, 0.0472, -1.621, 2.199, 0.0513, -2.148, 6.347, 0.9966, -3.732), 2),  # 2009-08-15, TAW 44 mm, DAP 93
    ((98.11, 61, 0.7332, 7.514, 5.509, 0, 1.537, 35.42, 21.78, 40.7, 2.71, 26.04, 0, 15.65, 0, 4.365, 34.68, 0, 9.673, -25.01, 69.16, 0, 19.29, -49.87, 0.3535, 0, -0.2549, 0.7049, 0, -0.5083, 1.033, 0, -0.7452, 1.329, 0, -0.9582, 2.414, 0.0229, -1.728), 2),  # 2009-09-19, TAW 98 mm, DAP 61
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
