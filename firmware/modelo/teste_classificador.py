# AgroEdge - teste rápido do classificador na BitDogLab.
# Copie classificador.py, agroedge_pesos.py e este arquivo para a placa e rode este.
# Mostra a classe de 6 dias reais do conjunto de teste (2009), o tempo de cada
# inferência e a memória livre.

import gc
import time
import classificador

# (features na ordem de classificador.FEATURES, classe esperada)
DIAS = (
    ((49, 0.732, 6.511, 4.766, 0, 13.28, 32.32, 20.17, 48.6, 2.8, 24.43, 0, 13.96, 0, 13.28, 30.84, 0, 38.38, 7.537, 60.7, 0, 63.16, 2.457), 0),  # 2009-09-02
    ((93, 0.6238, 4.808, 2.999, 0, 0, 30.42, 22.9, 70, 2.425, 20.42, 3, 8.761, 0.18, 0, 18.77, 46.17, 4.174, 31.57, 38.87, 122.1, 8.875, 92.07), 0),  # 2009-04-18
    ((35, 0.7015, 4.896, 3.434, 0, 0, 32.11, 21.15, 64.12, 2.294, 19.31, 3, 10.8, 0, 0, 25.49, 0, 16, -9.491, 50.21, 0.479, 32.14, -17.58), 1),  # 2009-08-19
    ((23, 0.6504, 4.978, 3.238, 0, 0, 30.42, 21.31, 58.3, 3.19, 15.7, 8, 9.362, 0.12, 0, 21.75, 1.51, 0, -20.24, 42.77, 1.673, 13.19, -27.91), 1),  # 2009-08-07
    ((42, 0.7207, 6.031, 4.346, 0, 0, 32.08, 21.28, 51.17, 3.175, 19.34, 2, 12.73, 0, 8.886, 29.86, 0, 8.886, -20.97, 55.35, 0, 8.886, -46.47), 2),  # 2009-08-26
    ((48, 0.7309, 6.492, 4.745, 0, 0, 32.67, 20.66, 53.47, 3.057, 24.16, 5, 14.01, 0, 0, 30.42, 0, 17.65, -12.78, 59.37, 0, 17.65, -41.73), 2),  # 2009-09-01
)

gc.collect()
print("memoria livre:", gc.mem_free(), "bytes")
acertos = 0
for x, esperada in DIAS:
    t0 = time.ticks_us()
    classe, saidas = classificador.classificar(x)
    dt = time.ticks_diff(time.ticks_us(), t0)
    acertos += classe == esperada
    print("%-17s esperado %-17s %6d us" % (classificador.CLASSES[classe],
                                          classificador.CLASSES[esperada], dt))
print("acertos: %d de %d" % (acertos, len(DIAS)))
