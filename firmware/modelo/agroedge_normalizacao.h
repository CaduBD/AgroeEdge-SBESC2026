// Gerado por modelo/03_treinar_mlp_int8.py - médias e desvios do conjunto de TREINO
#pragma once
#define AGROEDGE_N_FEATURES 23
// ordem: dap, kc, ETo, ETc, chuva, irrigacao, Tmax, Tmin, UR, vento, radiacao, dias_sem_irrigar, ETc_3d, chuva_3d, irrigacao_3d, ETc_7d, chuva_7d, irrigacao_7d, saldo_7d, ETc_14d, chuva_14d, irrigacao_14d, saldo_14d
static const float agroedge_media[AGROEDGE_N_FEATURES] = {54.000000f, 0.656120f, 5.611401f, 3.692077f, 1.480792f, 2.080177f, 32.126648f, 21.850164f, 59.577763f, 2.491442f, 21.186766f, 7.098706f, 10.994017f, 4.392018f, 6.174500f, 25.260426f, 10.080286f, 14.174993f, -1.005147f, 49.014317f, 19.909973f, 27.411909f, -1.692432f};
static const float agroedge_desvio[AGROEDGE_N_FEATURES] = {31.464266f, 0.071300f, 1.204047f, 0.912652f, 5.055694f, 5.730947f, 2.046458f, 1.488177f, 11.739520f, 0.754789f, 3.707079f, 13.498107f, 2.734867f, 11.099628f, 8.857765f, 6.799345f, 20.978611f, 13.043057f, 22.865618f, 15.327738f, 35.062565f, 21.503937f, 36.297894f};
static const float agroedge_in_scale = 0.04415027f;
static const int agroedge_in_zero_point = -49;
