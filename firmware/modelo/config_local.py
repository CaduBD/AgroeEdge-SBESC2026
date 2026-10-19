# AgroEdge - dados da área onde a placa está instalada. Edite na instalação.
# O modelo foi treinado com TAW de 25 a 140 mm; fora disso a resposta não é confiável.

import time

# água que o solo segura e a planta consegue usar (θCC − θPMP, em m³/m³)
# arenoso ~0,06-0,09 | franco-arenoso ~0,11 | argiloso ou Vertissolo ~0,15-0,18
AGUA_DISP = 0.11

# profundidade efetiva das raízes (m); com gotejamento, de 0,4 a 0,8
RAIZ_M = 0.6

# data da última poda (ano, mês, dia); atualize a cada poda
DATA_PODA = (2026, 1, 15)


def taw():
    """Água total disponível no solo, em mm (FAO-56, eq. 82)."""
    return 1000 * AGUA_DISP * RAIZ_M


def dap(hoje):
    """Dias após a poda. hoje = (ano, mês, dia)."""
    a = time.mktime(DATA_PODA + (0, 0, 0, 0, 0, 0))
    b = time.mktime(tuple(hoje) + (0, 0, 0, 0, 0, 0))
    return int((b - a) // 86400)


def kc(d):
    """Kc da uva Itália pelo DAP (Ávila Netto et al., 2000)."""
    return -8e-5 * d * d + 0.0089 * d + 0.488
