# AgroEdge - memória dos últimos 14 dias e montagem das 24 entradas do classificador.
# As contas são as mesmas do modelo/01_gerar_dataset.py (janelas de 3, 7 e 14 dias,
# dias sem irrigar, déficit), para a placa ver os dados como o modelo viu no treino.
# O estado fica num arquivo JSON na flash, para não se perder quando a placa reinicia.
#
# Uso, uma vez por dia, no fim do dia:
#   h = historico.Historico()                      # carrega o arquivo, se existir
#   x, deficit = h.fechar_dia(taw, dap, eto, chuva, irrigacao, tmax, tmin, ur, vento, radiacao)
#   classe, saidas = classificador.classificar(x)
#   h.salvar()

import json
import classificador
import config_local as cfg
import lamina

JANELA = 14


class Historico:
    def __init__(self, arquivo="historico.json"):
        self.arquivo = arquivo
        self.zerar()
        if arquivo:
            try:
                with open(arquivo) as f:
                    e = json.load(f)
                self.etc, self.chuva, self.irr = e["etc"], e["chuva"], e["irr"]
                self.ultima_irr, self.deficit, self.ultimo_dap = e["ultima_irr"], e["deficit"], e["ultimo_dap"]
            except (OSError, ValueError, KeyError):
                pass                     # primeiro uso ou arquivo estragado: começa do zero

    def zerar(self):
        """Começo de ciclo: na poda o solo está na capacidade de campo (irrigação de pré-poda)."""
        self.etc, self.chuva, self.irr = [], [], []
        self.ultima_irr = 0              # DAP da última irrigação (a poda conta como 0)
        self.deficit = 0.0
        self.ultimo_dap = -1

    def salvar(self):
        with open(self.arquivo, "w") as f:
            json.dump({"etc": self.etc, "chuva": self.chuva, "irr": self.irr,
                       "ultima_irr": self.ultima_irr, "deficit": self.deficit,
                       "ultimo_dap": self.ultimo_dap}, f)

    def fechar_dia(self, taw, dap, eto, chuva, irrigacao, tmax, tmin, ur, vento, radiacao):
        """Registra o dia que terminou e devolve (entradas do classificador, déficit em mm).
        irrigacao = mm aplicados hoje. Se o DAP voltou para trás, houve poda: recomeça o ciclo."""
        if dap <= self.ultimo_dap:
            self.zerar()
        self.ultimo_dap = dap
        kc = cfg.kc(dap)
        etc = kc * eto
        for lista, v in ((self.etc, etc), (self.chuva, chuva), (self.irr, irrigacao)):
            lista.append(v)
            if len(lista) > JANELA:
                lista.pop(0)
        if irrigacao > 0:
            self.ultima_irr = dap
        self.deficit = lamina.atualizar_deficit(self.deficit, etc, chuva, irrigacao, taw)

        v = {"TAW": taw, "dap": dap, "kc": kc, "ETo": eto, "ETc": etc, "chuva": chuva,
             "irrigacao": irrigacao, "Tmax": tmax, "Tmin": tmin, "UR": ur, "vento": vento,
             "radiacao": radiacao, "dias_sem_irrigar": dap - self.ultima_irr}
        for n in (3, 7, 14):
            v["ETc_%dd" % n] = sum(self.etc[-n:])
            v["chuva_%dd" % n] = sum(self.chuva[-n:])
            v["irrigacao_%dd" % n] = sum(self.irr[-n:])
        for n in (7, 14):
            v["saldo_%dd" % n] = v["chuva_%dd" % n] + v["irrigacao_%dd" % n] - v["ETc_%dd" % n]
        return [v[nome] for nome in classificador.FEATURES], self.deficit
