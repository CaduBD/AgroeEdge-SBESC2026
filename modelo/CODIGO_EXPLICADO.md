# Código explicado, função por função

Este arquivo percorre os três scripts do PC e os três arquivos MicroPython da placa, na ordem em que o código aparece. Para cada trecho: o código, o que ele faz e por quê. O `GUIA.md` dá a visão geral; aqui é o detalhe.

Bibliotecas que aparecem:
- **numpy (`np`)**: contas com vetores e números aleatórios.
- **pandas (`pd`)**: tabelas (`DataFrame`), como uma planilha dentro do Python.
- **scikit-learn (`sklearn`)**: MLP, SVM, Random Forest e as métricas.
- **TensorFlow / Keras (`tf`)**: treina a MLP final no PC. Na placa não entra TensorFlow: só MicroPython puro.

---

# Parte 1: `01_gerar_dataset.py`

## 1.1 Imports e caminhos
```python
import os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV_CLIMA = os.path.join(AQUI, "dados", "petrolina_clima_2000_2009.csv")
```
- `__file__` é o caminho deste script. `abspath` deixa o caminho completo, e `dirname` tira o nome do arquivo, sobrando só a pasta `modelo/`.
- `os.path.join` monta o caminho com a barra certa do sistema (`\` no Windows, `/` no Linux).
- Por que isso: o script funciona de qualquer pasta de onde você o rode, porque procura os dados ao lado dele, e não na pasta atual do terminal.

## 1.2 Parâmetros e constantes derivadas
```python
PODAS = [(1, 15), (7, 15)]
CICLO_DIAS = 109
ZR_M = 0.6
AGUA_DISP_M3M3 = 0.11
P_DEPLECAO = 0.35
KS_CRITICO = 0.5
CENARIOS_POR_CICLO = 8
SEMENTE = 42

TAW = 1000 * AGUA_DISP_M3M3 * ZR_M
RAW = P_DEPLECAO * TAW
DR_CRITICO = TAW - KS_CRITICO * (1 - P_DEPLECAO) * TAW
CLASSES = {0: "adequado", 1: "deficit_moderado", 2: "deficit_critico"}
```
- Nomes em MAIÚSCULAS são, por convenção do Python, constantes: valores que não mudam durante a execução. Ficam no topo para serem fáceis de achar e trocar.
- `PODAS` é uma lista de tuplas `(mês, dia)`.
- `TAW` multiplica por 1000 para converter metros de água em milímetros: 0,11 m³ de água por m³ de solo × 0,6 m de raiz = 0,066 m = **66 mm**.
- `DR_CRITICO` sai de isolar `Dr` na fórmula do Ks do FAO-56 (seção 1.6) com `Ks = 0,5`: dá **44,55 mm**.
- `CLASSES` é um dicionário que traduz o número da classe para texto, só para os `print`.

## 1.3 `kc_uva(dap)`
```python
def kc_uva(dap):
    return -8e-5 * dap**2 + 0.0089 * dap + 0.488
```
- `def` cria uma função; `dap` é o parâmetro de entrada.
- `-8e-5` é notação científica: −8 × 10⁻⁵ = −0,00008.
- `**` é potência em Python (`dap**2` = dap²).
- `return` devolve o valor. Ex.: `kc_uva(0)` = 0,488 e `kc_uva(55)` ≈ 0,73.

## 1.4 `carregar_clima()`
```python
def carregar_clima():
    df = pd.read_csv(CSV_CLIMA, parse_dates=["data"]).set_index("data").sort_index()
    completo = pd.date_range(df.index.min(), df.index.max(), freq="D")
    faltando = completo.difference(df.index)
    df = df.reindex(completo).interpolate()
    df["interpolado"] = df.index.isin(faltando).astype(int)
    return df
```
Linha por linha:
1. `pd.read_csv(...)` lê o CSV para uma tabela. `parse_dates=["data"]` faz a coluna `data` virar data de verdade, e não texto.
   - `.set_index("data")` usa a data como "nome" de cada linha, e assim dá para buscar `df.loc["2005-03-10"]`.
   - `.sort_index()` garante a ordem cronológica.
   - Encadear métodos com `.` assim é comum no pandas: cada um devolve uma tabela nova.
2. `pd.date_range(início, fim, freq="D")` cria a lista de **todos** os dias do período (3.653).
3. `completo.difference(df.index)` devolve os dias que estão na lista completa mas não no CSV: os 2 que faltam.
4. `df.reindex(completo)` cria linhas para esses dias, com valores vazios (`NaN`). `.interpolate()` preenche cada vazio com uma reta entre o dia anterior e o seguinte.
5. `df.index.isin(faltando)` devolve `True/False` para cada dia; `.astype(int)` converte para 1/0. É a coluna que marca os dias inventados.

## 1.5 `politica_aleatoria(rng)`
```python
def politica_aleatoria(rng):
    tipo = rng.choice(["gatilho", "turno_fixo"])
    if tipo == "gatilho":
        return dict(tipo=tipo,
                    gatilho=rng.uniform(0.3, 2.4) * RAW,
                    fracao=rng.uniform(0.3, 1.1),
                    falha=rng.uniform(0.0, 0.5))
    return dict(tipo=tipo,
                turno=int(rng.integers(1, 8)),
                lamina_fator=rng.uniform(0.2, 1.2),
                falha=rng.uniform(0.0, 0.5))
```
- `rng` é o gerador de números aleatórios criado no `main()`. Passá-lo como parâmetro, em vez de usar um global, deixa claro de onde vem a aleatoriedade.
- `rng.choice([...])` sorteia um item da lista.
- `rng.uniform(a, b)` sorteia um número real entre a e b. `rng.integers(1, 8)` sorteia um inteiro de 1 a **7** (o 8 não entra).
- `dict(chave=valor, ...)` cria um dicionário. A função devolve as "regras" do produtor, por exemplo `{"tipo": "gatilho", "gatilho": 31.2, "fracao": 0.8, "falha": 0.1}`.
- Se o `if` for verdadeiro, o primeiro `return` encerra a função; senão, cai no segundo `return`. Por isso não precisa de `else`.

## 1.6 `simular_ciclo(clima, inicio, pol, rng, cenario_id)`
É a função mais importante. Ela simula um ciclo de 109 dias para um produtor.

```python
    dias = clima.loc[inicio: inicio + pd.Timedelta(days=CICLO_DIAS - 1)]
    dr = 0.0
    ultimo = 0
    linhas = []
```
- `clima.loc[a:b]` recorta as linhas entre duas datas, **incluindo** as duas pontas. `pd.Timedelta(days=108)` é "108 dias depois", então são 109 dias.
- `dr = 0.0`: o ciclo começa com o solo cheio (supõe-se uma irrigação antes da poda).
- `ultimo` guarda o último dia em que houve irrigação.
- `linhas` é uma lista vazia que vai receber um dicionário por dia.

```python
    for dap, (data, c) in enumerate(dias.iterrows()):
```
- `dias.iterrows()` percorre a tabela linha por linha e devolve pares `(data, linha)`.
- `enumerate(...)` acrescenta um contador que começa em 0, que aqui é o próprio **DAP**.
- `(data, c)` desempacota o par: `c` é a linha do dia, e `c.ETo`, `c.Precipitacao` etc. são as colunas.

```python
        kc = kc_uva(dap)
        etc = kc * c.ETo
        irr = 0.0
        if rng.random() >= pol["falha"]:
            if pol["tipo"] == "gatilho" and dr > pol["gatilho"]:
                irr = pol["fracao"] * dr
            elif pol["tipo"] == "turno_fixo" and dap % pol["turno"] == 0:
                irr = pol["lamina_fator"] * etc * pol["turno"]
```
- `rng.random()` sorteia um número entre 0 e 1. Se `falha = 0,2`, o produtor irriga em 80% dos dias em que deveria.
- **Gatilho:** se o solo já gastou mais que o gatilho, repõe uma fração do que falta (`fracao × dr`).
- **Turno fixo:** `dap % turno == 0` é "o resto da divisão é zero", ou seja, dia 0, N, 2N... Nesses dias aplica `fator × ETc × N`, a água de N dias de uma vez.
- A decisão usa o `dr` de **ontem**, porque o produtor decide de manhã.

```python
        ks = 1.0 if dr <= RAW else max(0.0, (TAW - dr) / ((1 - P_DEPLECAO) * TAW))
        etc_aj = ks * etc
        dr = dr - c.Precipitacao - irr + etc_aj
        dr = min(max(dr, 0.0), TAW)
```
- `x if condição else y` é um `if` em uma linha.
- A fórmula do Ks é a equação 84 do FAO-56. Com `dr = RAW` dá 1; com `dr = TAW` dá 0; no meio cai em linha reta. O `max(0.0, ...)` evita número negativo.
- `etc_aj` é o consumo real: a planta com sede consome menos que o ETc.
- A atualização do `dr` é a equação 85: gasto aumenta `dr`, chuva e irrigação diminuem.
- `min(max(dr, 0), TAW)` "prende" o valor entre 0 e 66. Abaixo de 0 seria solo mais que cheio (o excesso drena); acima de 66 seria solo mais seco que o ponto de murcha.

```python
        if irr > 0:
            ultimo = dap
        if dr <= RAW:
            classe = 0
        elif dr <= DR_CRITICO:
            classe = 1
        else:
            classe = 2
        linhas.append(dict(data=data, cenario=cenario_id, dap=dap, kc=kc, ...,
                           dias_sem_irrigar=dap - ultimo, ..., Dr=dr, classe=classe))
    return linhas
```
- O rótulo é decidido com o `dr` do **fim do dia**.
- `linhas.append(...)` adiciona um dicionário com tudo do dia. Cada chave vira uma coluna depois.
- `dias_sem_irrigar = dap - ultimo`. Detalhe: se o produtor nunca irrigou, conta desde a poda.
- `Dr` é salvo para conferência, mas **não** entra como feature (ver Parte 2).

## 1.7 `adicionar_janelas(df)`
```python
def adicionar_janelas(df):
    g = df.groupby(["cenario"], sort=False)
    for n in (3, 7, 14):
        for col in ("ETc", "chuva", "irrigacao"):
            df[f"{col}_{n}d"] = g[col].transform(lambda s: s.rolling(n, min_periods=1).sum())
    for n in (7, 14):
        df[f"saldo_{n}d"] = df[f"chuva_{n}d"] + df[f"irrigacao_{n}d"] - df[f"ETc_{n}d"]
    return df
```
- `df.groupby("cenario")` separa a tabela em grupos, um por ciclo × produtor. Isso impede que a soma de 14 dias de um ciclo "vaze" para o ciclo seguinte.
- `f"{col}_{n}d"` é uma **f-string**: monta o texto trocando `{col}` e `{n}` pelos valores. Ex.: `"chuva_7d"`.
- `s.rolling(n).sum()` é a **soma móvel**: para cada dia, soma os n últimos (incluindo hoje). `min_periods=1` faz os primeiros dias do ciclo somarem o que existe, em vez de dar vazio.
- `lambda s: ...` é uma função sem nome, de uma linha. `transform` aplica essa função em cada grupo e devolve o resultado alinhado com a tabela original.
- O laço duplo cria 9 colunas (3 janelas × 3 variáveis), e o segundo laço cria mais 2 de saldo.

## 1.8 `main()`
```python
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
```
- `np.random.default_rng(42)` cria o gerador com semente fixa: rodando de novo, sai exatamente o mesmo dataset.
- `set(clima.index.year)` pega os anos sem repetir; `sorted` põe em ordem.
- `continue` pula para a próxima volta do laço. Aqui ele pularia um ciclo que terminasse depois do fim dos dados (não acontece com 15/07, mas protege se alguém mudar a data).
- `for _ in range(8)`: o `_` é um nome de variável que significa "não vou usar o contador".
- `linhas += lista` junta as listas. `cid` numera cada cenário de 0 a 159.

```python
    df = adicionar_janelas(pd.DataFrame(linhas))
    df["ano"] = df["data"].dt.year
    df["split"] = np.where(df.ano <= 2006, "treino", np.where(df.ano <= 2008, "validacao", "teste"))
    df.to_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"), index=False, float_format="%.4f")
```
- `pd.DataFrame(linhas)` transforma a lista de dicionários em tabela.
- `.dt.year` extrai o ano de cada data.
- `np.where(condição, se_sim, se_não)` é um `if` aplicado na coluna inteira de uma vez. Aninhado, dá três valores.
- `to_csv(..., index=False)` salva sem a coluna de índice; `float_format="%.4f"` grava 4 casas decimais, o que deixa o arquivo menor.

```python
    tab = pd.crosstab(df.split, df.classe.map(CLASSES), normalize="index").round(3)
```
- `crosstab` conta quantos dias há de cada classe em cada split. `normalize="index"` transforma em proporção por linha.

```python
if __name__ == "__main__":
    main()
```
- Só roda o `main()` quando você executa o arquivo direto (`python 01_gerar_dataset.py`). Se outro script importar este, nada roda sozinho. O script 03 usa isso para importar o 02.

---

# Parte 2: `02_comparar_modelos.py`

## 2.1 Imports
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
```
No scikit-learn, todo modelo tem a mesma "cara": `modelo.fit(X, y)` treina e `modelo.predict(X)` responde. Por isso dá para comparar três algoritmos com o mesmo código.

## 2.2 `FEATURES` e `NOMES`
```python
FEATURES = ["dap", "kc", "ETo", "ETc", "chuva", "irrigacao", "Tmax", "Tmin", "UR",
            "vento", "radiacao", "dias_sem_irrigar",
            "ETc_3d", ..., "saldo_14d"]
```
A lista define **quais colunas** entram no modelo e **em que ordem**. O firmware precisa montar o vetor exatamente nessa ordem. `Dr`, `classe`, `data`, `cenario`, `split` e `interpolado` ficam de fora: são resposta ou controle, não entrada.

## 2.3 Funções de memória
```python
def memoria_mlp(m):
    return sum(w.size + b.size for w, b in zip(m.coefs_, m.intercepts_)) * 4
```
- `m.coefs_` é a lista das matrizes de pesos de cada camada; `m.intercepts_` é a lista dos bias.
- `zip(a, b)` anda nas duas listas juntas, aos pares.
- `w.size` é o número de elementos da matriz. A soma de tudo × 4 bytes (um float32) dá o tamanho.

```python
def memoria_svm(m):
    return (m.support_vectors_.size + m.dual_coef_.size + m.intercept_.size) * 4
```
- O SVM com kernel RBF precisa guardar os **vetores de suporte**: dias de treino inteiros (23 números cada) que definem a fronteira entre as classes. Com 3.512 vetores, são 3.512 × 23 × 4 bytes, e é por isso que ele fica grande.

```python
def memoria_rf(m):
    total = 0
    for arv in m.estimators_:
        t = arv.tree_
        internos = int((t.children_left != -1).sum())
        folhas = t.node_count - internos
        total += internos * 10 + folhas * 1
    return total
```
- `m.estimators_` é a lista de árvores. Em cada árvore, `children_left` tem o filho esquerdo de cada nó, e −1 indica que é folha.
- `(t.children_left != -1).sum()` conta os `True`, que são os nós internos.
- Cada nó interno precisa de feature (2 B), limiar (4 B) e dois filhos (2 × 2 B) = 10 B. Cada folha guarda só a classe (1 B). É uma **estimativa** do tamanho da árvore guardada de forma compacta.

## 2.4 `main()`: preparação
```python
    df = pd.read_csv(os.path.join(AQUI, "dados", "dataset_uva.csv"))
    tr, va, te = (df[df.split == s] for s in ("treino", "validacao", "teste"))
    sc = StandardScaler().fit(tr[FEATURES])
    Xtr, Xva, Xte = (sc.transform(d[FEATURES]) for d in (tr, va, te))
    ytr, yva, yte = tr.classe.values, va.classe.values, te.classe.values
```
- `df[df.split == "treino"]` filtra só as linhas do treino. O `( ... for s in ...)` faz isso para os três grupos, e `tr, va, te =` recebe os três resultados.
- `StandardScaler().fit(...)` calcula média e desvio **só no treino**. `.transform(...)` aplica `(x − média)/desvio` em cada grupo.
- Convenção: `X` (maiúsculo) são as entradas e `y` (minúsculo) a resposta. `.values` tira a coluna do pandas e devolve um vetor numpy.

## 2.5 `main()`: os candidatos
```python
    candidatos = {
        "MLP": [MLPClassifier(hidden_layer_sizes=h, max_iter=600, early_stopping=True,
                              random_state=0) for h in [(16,), (32,), (16, 8), (32, 16)]],
        "SVM (RBF)": [SVC(C=c, gamma="scale", class_weight="balanced") for c in (1, 10)],
        "Random Forest": [RandomForestClassifier(n_estimators=n, max_depth=d, class_weight="balanced",
                                                 random_state=0, n_jobs=-1)
                          for n, d in [(20, 8), (50, 10), (100, None)]],
    }
```
- Um dicionário em que cada valor é uma lista de modelos ainda não treinados, criada com **list comprehension** (`[expressão for x in lista]`).
- **MLP:**
  - `hidden_layer_sizes=(16, 8)` são duas camadas escondidas, de 16 e de 8 neurônios.
  - `early_stopping=True` para de treinar quando a validação interna para de melhorar.
  - `random_state=0` fixa os pesos iniciais.
- **SVM:** `C` controla o quanto ele aceita errar no treino (C maior = fronteira mais justa). `gamma="scale"` ajusta o alcance do kernel automaticamente.
- **RF:** `n_estimators` é o número de árvores e `max_depth` a profundidade máxima (`None` = sem limite). `n_jobs=-1` usa todos os núcleos do PC.

## 2.6 `main()`: escolher, testar e medir
```python
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
```
- `.items()` percorre o dicionário em pares `(chave, valor)`.
- Para cada configuração: treina no treino, prevê a validação e calcula o F1 macro. Guarda numa tupla `(f1, modelo)` o melhor até agora.
- O mesmo `m.fit(Xtr, ytr)` serve para os três. O desbalanceamento (poucos dias "crítico") é tratado dentro do próprio modelo: SVM e RF recebem `class_weight="balanced"`, que dá às classes raras um peso maior, inversamente proporcional ao número de dias de cada uma.
- O `MLPClassifier` do scikit-learn não tem `class_weight`. Por isso, aqui a MLP compete sem essa ajuda, e mesmo assim fica perto dos outros. Na MLP final (Parte 3, no Keras) o peso das classes é usado.

```python
        t0 = time.perf_counter()
        p = m.predict(Xte)
        dt = (time.perf_counter() - t0) / len(Xte) * 1e6
```
- Mede o tempo de previsão no teste e divide pelo número de dias, em microssegundos (`1e6`). É o tempo **no PC**, só para comparar os três entre si. Não vale para o RP2040.

```python
        mem = {"MLP": memoria_mlp, "SVM (RBF)": memoria_svm, "Random Forest": memoria_rf}[nome](m)
        rec = recall_score(yte, p, average=None)
```
- O dicionário guarda **funções**; `[nome]` escolhe a certa e `(m)` a chama. É um jeito curto de trocar um `if/elif`.
- `average=None` devolve a revocação de cada classe separada: `rec[0]`, `rec[1]`, `rec[2]`.

```python
        linhas.append(dict(modelo=nome, config=str(cfg), acuracia=..., ...))
        matrizes[nome] = confusion_matrix(yte, p)
```
- Cada modelo vira uma linha da tabela final. `confusion_matrix` devolve a matriz 3 × 3.

## 2.7 Saída
```python
    res = pd.DataFrame(linhas)
    print(res.round(3).to_string(index=False))
    with open(os.path.join(AQUI, "resultados_comparacao.md"), "w") as f:
        f.write(res.round(3).to_markdown(index=False) + "\n\n")
        ...
```
- `with open(..., "w") as f:` abre o arquivo para escrita e o fecha sozinho no fim do bloco, mesmo se der erro.
- `to_markdown` (precisa da biblioteca `tabulate`) gera a tabela no formato que o GitHub mostra bonito.

---

# Parte 3: `03_treinar_mlp.py`

## 3.1 Imports, caminhos e o script 02
```python
AQUI = os.path.dirname(os.path.abspath(__file__))
PASTA_FW = os.path.join(AQUI, "..", "firmware", "modelo")
sys.path.insert(0, AQUI)
FEATURES = import_module("02_comparar_modelos").FEATURES
CLASSES = ("adequado", "deficit_moderado", "deficit_critico")
NOMES = ["adequado", "moderado", "critico"]
OCULTA = 16
```
- `PASTA_FW` aponta para `firmware/modelo/`: o `..` sobe uma pasta, de `modelo/` para a raiz do repositório.
- Queremos reaproveitar a lista `FEATURES` do 02, para não haver duas listas que podem ficar diferentes.
- Mas `import 02_comparar_modelos` é erro de sintaxe, porque nome de módulo não pode começar com número. `import_module("...")` recebe o nome como texto e contorna isso.
- `sys.path.insert(0, pasta)` avisa o Python para procurar módulos nessa pasta.
- Graças ao `if __name__ == "__main__"` do 02, importar não roda o treino dele.
- `CLASSES` é uma **tupla** (parênteses): como uma lista, mas não pode ser alterada. Vai para o arquivo da placa com os nomes das classes.
- `NOMES` são os rótulos curtos usados só na matriz de confusão do relatório.

## 3.2 `lista_f(v)`: número do numpy para texto do MicroPython
```python
def lista_f(v):
    nums = ", ".join(f"{x:.8g}" for x in np.asarray(v, dtype=np.float32).ravel())
    return f'array("f", [{nums}])'
```
- `np.asarray(v, dtype=np.float32)` garante que os números estão em float de 32 bits, o mesmo tamanho do float do MicroPython no RP2040.
- `.ravel()` "achata" uma matriz em uma fila só. A matriz `W1` (23 × 16) vira um vetor de 368 números, linha atrás de linha.
- `f"{x:.8g}"` escreve o número com até 8 algarismos significativos. Um float32 guarda cerca de 7, então 8 garante que nada se perde ao virar texto. O `g` escolhe sozinho entre `0.00123` e `1.23e-05`, o mais curto.
- O resultado é um texto como `array("f", [0.12, -0.5, ...])`, que é código Python válido. Quando a placa importar o arquivo, isso cria o vetor.

## 3.3 `exportar(...)`: escrever o `agroedge_pesos.py`
```python
def exportar(media, desvio, W1, B1, W2, B2):
    os.makedirs(PASTA_FW, exist_ok=True)
    caminho = os.path.join(PASTA_FW, "agroedge_pesos.py")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("# Gerado por modelo/03_treinar_mlp.py - não editar à mão\n")
        f.write("from array import array\n\n")
        f.write(f"N_FEATURES = {len(FEATURES)}\nN_OCULTA = {OCULTA}\nN_CLASSES = {len(CLASSES)}\n\n")
        f.write("FEATURES = (" + ", ".join(f'"{n}"' for n in FEATURES) + ")\n")
        ...
        f.write(f"MEDIA = {lista_f(media)}\nDESVIO = {lista_f(desvio)}\n\n")
        f.write(f"W1 = {lista_f(W1)}\nB1 = {lista_f(B1)}\n\n")
        f.write(f"W2 = {lista_f(W2)}\nB2 = {lista_f(B2)}\n")
    return caminho
```
- `os.makedirs(..., exist_ok=True)` cria a pasta se ela não existir e não reclama se já existir.
- `encoding="utf-8"` garante que os acentos dos comentários saiam certos também no Windows.
- O script **escreve um programa Python**: cada `f.write` é uma linha do arquivo que vai para a placa. Por isso o arquivo começa com `from array import array`, para o `array("f", ...)` funcionar lá.
- `", ".join(f'"{n}"' for n in FEATURES)` põe aspas em volta de cada nome e vírgulas entre eles: `"dap", "kc", ...`.

## 3.4 Normalização manual
```python
    media = tr[FEATURES].mean().values.astype(np.float32)
    desvio = tr[FEATURES].std(ddof=0).values.astype(np.float32)
    norm = lambda d: ((d[FEATURES].values - media) / desvio).astype(np.float32)
    Xtr, Xva, Xte = norm(tr), norm(va), norm(te)
```
- Aqui a normalização é feita "à mão", e não com `StandardScaler`, porque precisamos dos números `media` e `desvio` para escrever no arquivo da placa.
- `ddof=0` usa o desvio populacional (divide por N), o mesmo do `StandardScaler`. O padrão do pandas é dividir por N−1.
- `media` tem 23 valores. O numpy subtrai cada um da coluna certa automaticamente (isso se chama *broadcasting*).

## 3.5 Pesos das classes
```python
    pesos = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr)))
```
- `compute_class_weight` devolve um peso por classe: `n_dias / (3 × n_dias_da_classe)`. A classe crítica, com poucos dias, ganha o maior peso.
- `enumerate` + `dict` transformam o vetor em `{0: p0, 1: p1, 2: p2}`, o formato que o Keras pede.

## 3.6 A rede no Keras e o treino
```python
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(OCULTA, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200, batch_size=64,
               class_weight=pesos, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    p_keras = modelo.predict(Xte, verbose=0).argmax(1)
```
- `Sequential` é uma rede em que os dados passam por uma camada depois da outra.
- `Input(shape=(23,))` diz que cada exemplo tem 23 números.
- `Dense(16, activation="relu")`: cada um dos 16 neurônios calcula `relu(w · x + b)`. Isso dá 23 × 16 pesos + 16 bias.
- `Dense(3, activation="softmax")`: 3 saídas transformadas em probabilidades.
- `compile` configura o treino:
  - **Adam** é o algoritmo que ajusta os pesos, com passo (taxa de aprendizado) de 0,001.
  - **loss** é o erro que ele tenta diminuir. *Sparse categorical crossentropy* é o erro padrão para classificação quando o rótulo é um número inteiro (0, 1, 2).
- `fit` treina:
  - **época** = uma passada por todo o treino;
  - **batch** = os pesos são ajustados a cada 64 dias;
  - `EarlyStopping` para o treino se o erro na validação não melhorar por 15 épocas seguidas e volta para a melhor época. As 200 épocas são só um teto.
- `predict` devolve uma matriz (dias × 3) de probabilidades. `.argmax(1)` pega, em cada linha, o índice da maior, que é a classe prevista.

## 3.7 Tirar os pesos do Keras
```python
    (W1, B1), (W2, B2) = (camada.get_weights() for camada in modelo.layers)
    caminho = exportar(media, desvio, W1, B1, W2, B2)
```
- `modelo.layers` são as duas camadas `Dense`. `get_weights()` devolve `[pesos, bias]` de cada uma.
- `W1` tem formato (23, 16): `W1[i][j]` liga a entrada `i` ao neurônio `j`. `W2` tem formato (16, 3).
- A linha desempacota os dois pares de uma vez: `(W1, B1)` da primeira camada, `(W2, B2)` da segunda.

## 3.8 Conferir com o próprio `classificador.py`
```python
    sys.path.insert(0, PASTA_FW)
    clf = import_module("classificador")
    p_mcu = np.array([clf.classificar(list(x))[0] for x in te[FEATURES].values.astype(np.float32)])
```
- Importa o mesmo `classificador.py` que vai para a placa, que por sua vez importa o `agroedge_pesos.py` recém-gerado.
- Passa os dias de teste **sem normalizar** (`te[FEATURES].values`), do jeito que o firmware vai passar. A normalização acontece dentro do `classificar`.
- `list(x)` converte a linha do numpy em lista comum, como na placa. `[0]` pega só a classe do par `(classe, saidas)`.
- Se `p_mcu` for igual a `p_keras`, a exportação está certa: a placa vai dar a mesma resposta que o Keras.

## 3.9 O relatório `resultados_mlp.md`
```python
    n_param = W1.size + B1.size + W2.size + B2.size
    ...
        for nome, p in (("Keras (float32)", p_keras), ("classificador.py (MicroPython)", p_mcu)):
            rec = recall_score(yte, p, average=None)
            f.write(f"| {nome} | {accuracy_score(yte, p):.3f} | ... |\n")
        f.write(f"\nConcordância Keras × classificador.py: {(p_keras == p_mcu).mean():.4f}.\n")
```
- `p_keras == p_mcu` compara os dois vetores posição a posição e dá `True/False`. `.mean()` de `True/False` é a fração de `True`: 1,0000 quer dizer 100% iguais.
- `n_param` dá 435. Somando as 46 médias e desvios e multiplicando por 4 bytes, dá o tamanho dos vetores na placa (1.924 bytes).

---

# Parte 4: `firmware/modelo/` (MicroPython, roda na BitDogLab)

## 4.1 `agroedge_pesos.py` (gerado)
```python
# Gerado por modelo/03_treinar_mlp.py - não editar à mão
from array import array

N_FEATURES = 23
N_OCULTA = 16
N_CLASSES = 3

FEATURES = ("dap", "kc", "ETo", ...)
CLASSES = ("adequado", "deficit_moderado", "deficit_critico")

MEDIA = array("f", [54, 0.65612018, ...])
DESVIO = array("f", [31.464266, 0.0713, ...])
W1 = array("f", [...])   # 368 números
B1 = array("f", [...])   # 16
W2 = array("f", [...])   # 48
B2 = array("f", [...])   # 3
```
- É um módulo só de dados: nenhuma função, só constantes.
- **Por que `array("f")` e não lista comum:** no MicroPython, cada float de uma lista é um objeto separado na memória, com um gasto extra por número. O `array("f")` guarda os números colados, 4 bytes cada, como um vetor em C. Para 481 números, são cerca de 1,9 KB.
- **Matriz achatada:** o MicroPython não tem matriz de duas dimensões, então `W1` é guardado como uma fila de 23 × 16 = 368 números, linha após linha. O peso que liga a entrada `i` ao neurônio `j` fica na posição `i * 16 + j`. Exemplo: entrada 2, neurônio 5 → posição 2 × 16 + 5 = 37.
- `MEDIA[0] = 54` é a média do `dap` (dias 0 a 108 → média 54), e assim por diante na ordem de `FEATURES`.

## 4.2 `classificador.py` (escrito à mão)
```python
import agroedge_pesos as P

FEATURES = P.FEATURES
CLASSES = P.CLASSES
```
- `import ... as P` dá um apelido curto ao módulo dos pesos: `P.W1` em vez de `agroedge_pesos.W1`.
- `FEATURES` e `CLASSES` são repassados para quem usa o classificador não precisar importar os pesos.

```python
def classificar(x):
    if len(x) != P.N_FEATURES:
        raise ValueError("esperava %d features" % P.N_FEATURES)
```
- Confere se chegaram 23 números. Se não, `raise` interrompe com uma mensagem de erro clara, em vez de dar um resultado errado sem avisar.
- `"%d" % valor` é a formatação de texto antiga do Python, que o MicroPython também aceita.

```python
    z = [(x[i] - P.MEDIA[i]) / P.DESVIO[i] for i in range(P.N_FEATURES)]
```
- **Passo 1, normalização:** a mesma conta do `norm` do script 03, feita para cada uma das 23 entradas com uma list comprehension.

```python
    h = []
    for j in range(P.N_OCULTA):
        s = P.B1[j]
        for i in range(P.N_FEATURES):
            s += z[i] * P.W1[i * P.N_OCULTA + j]
        h.append(s if s > 0.0 else 0.0)
```
- **Passo 2, camada escondida.** Para cada neurônio `j` (16 voltas):
  - começa a soma `s` pelo bias do neurônio;
  - soma `entrada × peso` para as 23 entradas (é o produto escalar `z · W1[:, j]`);
  - aplica a **ReLU**: se a soma deu negativa, vira 0.
- `h` termina com 16 valores, as "ativações" da camada escondida.
- São 16 × 23 = 368 multiplicações. É exatamente o que o `Dense(16, activation="relu")` do Keras calcula.

```python
    saidas = []
    melhor = 0
    for k in range(P.N_CLASSES):
        s = P.B2[k]
        for j in range(P.N_OCULTA):
            s += h[j] * P.W2[j * P.N_CLASSES + k]
        saidas.append(s)
        if s > saidas[melhor]:
            melhor = k
    return melhor, saidas
```
- **Passo 3, camada de saída.** Igual ao passo 2, agora com 3 neurônios (um por classe) e 16 entradas. Não tem ReLU.
- **Passo 4, argmax:** `melhor` guarda o índice da maior saída vista até agora. Na primeira volta (`k = 0`), `saidas[0]` é comparado com ele mesmo, e `melhor` continua 0.
- **Por que não calcular o softmax:** o softmax é `e^s / soma(e^s)`. Como a exponencial só cresce, a maior saída antes do softmax continua sendo a maior depois. Pular essa parte economiza 3 exponenciais, que são caras sem FPU.
- `return melhor, saidas` devolve uma tupla. Quem chama pode fazer `classe, saidas = classificar(x)`. As `saidas` servem para mandar ao dashboard, por exemplo para mostrar o quanto o modelo está "seguro".

## 4.3 `teste_classificador.py`
```python
import gc
import time
import classificador

DIAS = (
    ((49, 0.732, 6.511, ...), 0),  # 2009-09-02
    ...
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
```
- `DIAS` é uma tupla de pares `(features, classe_esperada)`: 6 dias reais de 2009, 2 de cada classe.
- `gc` é o coletor de lixo (*garbage collector*). `gc.collect()` libera a memória que não é mais usada, e `gc.mem_free()` mostra quanto sobrou. Isso é específico do MicroPython.
- `time.ticks_us()` lê um relógio em microssegundos. `time.ticks_diff(fim, início)` faz a subtração do jeito certo mesmo quando o contador "dá a volta" (passa do valor máximo e recomeça do zero). São funções do MicroPython; no Python normal não existem.
- `acertos += classe == esperada`: a comparação dá `True` (vale 1) ou `False` (vale 0), então soma 1 a cada acerto.
- `"%-17s"` escreve o texto alinhado à esquerda em 17 colunas, para a tabela sair alinhada no terminal.
- **Resultado conferido no PC**, num MicroPython compilado com float de 32 bits como o do RP2040: 6 de 6 acertos. O tempo mostrado no PC (~150 µs) não vale para a placa, que é bem mais lenta. Esse é o número que vocês devem medir na BitDogLab.

## 4.4 Como usar no firmware
```python
import classificador

x = montar_features()   # vocês escrevem: sensores + histórico de 14 dias
classe, saidas = classificador.classificar(x)
print(classificador.CLASSES[classe])
```
- `montar_features()` tem que devolver as 23 entradas **na ordem de `classificador.FEATURES`**, sem normalizar. Uma troca de ordem não dá erro: dá uma resposta errada sem aviso, então vale conferir com o `teste_classificador.py` sempre que mexer nisso.
- Para copiar para a placa, use o Thonny ou o `mpremote cp classificador.py agroedge_pesos.py :` e rode o `teste_classificador.py` primeiro.
