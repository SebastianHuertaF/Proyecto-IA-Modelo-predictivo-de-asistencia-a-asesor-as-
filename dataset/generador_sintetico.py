"""
Generador de datos sintéticos — Modelo predictivo de asistencia a asesorías
IA 1INF24, Grupo 6

Uso:
    python generador_sintetico.py respuestas.csv carpeta_salida

Entrada: el CSV de la hoja "Respuestas de formulario 1" (Archivo > Descargar > CSV).

Qué hace (sigue la Sec. C del informe):
  1. Convierte las respuestas reales al formato del modelo (una fila por
     estudiante × escenario × modalidad) con las 13 variables de la Tabla I.
  2. Divide a los estudiantes REALES en dos grupos que no se mezclan:
       - calibración (2/3): lo único que ve el generador.
       - prueba real (1/3): nunca se usa para calibrar; es la prueba final.
  3. Calibra el generador SOLO con la semilla de calibración:
       - perfiles: cópula gaussiana con marginales empíricas (como en [3]);
       - horario: probabilidad de estar ocupado y de que la franja "acomode",
         por día y franja;
       - coeficientes βj de la ec. (3) por regresión logística regularizada,
         con los signos del criterio experto, cercanía como e^(-d/τ) y su
         efecto amplificado cuando el promedio es bajo;
       - efecto aleatorio por estudiante (σ_u) y β0 calibrados por simulación
         para reproducir la tasa global y la variación entre estudiantes.
  4. Genera estudiantes y sesiones sintéticas y etiqueta y ~ Bernoulli(p).
  5. Particiona sin fuga de información:
       - por estudiante (nadie aparece en dos particiones);
       - con combinaciones de sesión RESERVADAS que no existen en entrenamiento,
         para comprobar que el modelo generaliza y no memoriza.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

SEMILLA = 2026
rng = np.random.default_rng(SEMILLA)

# ------------------------------------------------------------------ constantes
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado"]
DIAS_FORM = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
FRANJAS = ["manana", "tarde", "noche"]
FRANJAS_FORM = ["Mañana", "Tarde", "Noche"]
OPCION_VALOR = {"Sí, seguro": 4, "Probablemente sí": 3, "Probablemente no": 2, "No": 1}
VECES = {"Es la primera vez": 1, "Es la segunda vez": 2, "Tres veces o más": 3}
SIN_HIST = "Nunca me han ofrecido asesorías"

# Escenarios del formulario: tipo de franja, días a la evaluación, dificultad
ESCENARIOS = {
    "E1": ("acomoda", 1, 4), "E2": ("ocupada", 1, 2), "E3": ("libre", 4, 5),
    "E4": ("acomoda", 4, 1), "E5": ("ocupada", 7, 3), "E6": ("libre", 7, 2),
    "E7": ("acomoda", 14, 5), "E8": ("ocupada", 14, 4), "E9": ("libre", 30, 1),
    "E10": ("acomoda", 30, 3),
}
TIPO = {"ocupada": (1, 0), "libre": (0, 0), "acomoda": (0, 1)}  # -> (cruce, dispuesto)

# Tramos de plazo (coinciden con las opciones del formulario)
def tramo_plazo(d):
    if d <= 1:
        return "manana"
    if d <= 5:
        return "3a5dias"
    if d <= 10:
        return "1semana"
    if d <= 21:
        return "2semanas"
    return "1mes"

# Signos del criterio experto (Sec. C): +1 aumenta la asistencia, -1 la reduce
SIGNO_EXPERTO = {
    "asistencia_hist": +1, "carga_cursos": -1, "veces_llevado_curso": +1,
    "uso_recursos_alternativos": -1, "cruce_horario": -1, "dispuesto_franja": +1,
    "dificultad_tema": +1, "urgencia": +1, "urgencia_x_promedio_bajo": +1,
    "traslado_x_presencial": -1,
}
# Efectos de día y franja: la semilla no los identifica (el estudiante responde
# por TIPO de franja), así que se fijan con criterio experto y se documentan.
EFECTO_FRANJA = {"manana": 0.0, "tarde": 0.0, "noche": -0.25}
EFECTO_DIA = {d: 0.0 for d in DIAS} | {"viernes": -0.15, "sabado": -0.35}
SIGMA_EPS = 0.5  # ruido ε de la ec. (3), por sesión


# ================================================================ 1. semilla
def leer_semilla(ruta):
    raw = pd.read_csv(ruta)
    cols = list(raw.columns)
    def col(prefijo):
        return next(c for c in cols if c.startswith(prefijo))
    c_prom = col("Promedio ponderado")
    c_carga = col("Cursos matriculados")
    c_hist = col("De las asesorías")
    c_veces = next(c for c in cols if "veces estás llevando" in c)
    c_tras = next(c for c in cols if "minutos te toma" in c)
    c_rec = next(c for c in cols if "con qué frecuencia" in c)
    c_ocu = [next(c for c in cols if "actividad fija" in c and c.endswith(f"[{d}]")) for d in DIAS_FORM]
    c_aco = [next(c for c in cols if "te acomodaría" in c and c.endswith(f"[{d}]")) for d in DIAS_FORM]
    c_esc = {(e, m): next(c for c in cols if c.startswith(e + ".") and c.endswith(f"[{m}]"))
             for e in ESCENARIOS for m in ("Presencial", "Virtual")}

    def matriz(fila, cs):
        m = np.zeros((6, 3), dtype=int)
        for i, c in enumerate(cs):
            marcadas = [x.strip() for x in str(fila[c]).split(",")] if pd.notna(fila[c]) else []
            for j, f in enumerate(FRANJAS_FORM):
                m[i, j] = int(f in marcadas)
        return m

    perfiles, horarios, filas = [], {}, []
    for k, r in raw.iterrows():
        sid = f"real_{k + 1:03d}"
        sin_hist = int(r[c_hist] == SIN_HIST)
        p = {
            "id_estudiante": sid,
            "promedio_previo": float(str(r[c_prom]).replace(",", ".")),
            "asistencia_hist": np.nan if sin_hist else float(str(r[c_hist]).replace("%", "")) / 100,
            "sin_historial": sin_hist,
            "carga_cursos": int(r[c_carga]),
            "veces_llevado_curso": VECES[r[c_veces]],
            "tiempo_traslado": float(r[c_tras]),
            "uso_recursos_alternativos": int(r[c_rec]),
        }
        ocu, aco = matriz(r, c_ocu), matriz(r, c_aco)
        aco = aco * (1 - ocu)  # "acomoda" solo tiene sentido si está libre
        perfiles.append(p)
        horarios[sid] = (ocu, aco)

        # franjas reales del estudiante por tipo, para asignar dia/franja
        franjas = {"ocupada": [], "libre": [], "acomoda": []}
        for i in range(6):
            for j in range(3):
                t = "ocupada" if ocu[i, j] else ("acomoda" if aco[i, j] else "libre")
                franjas[t].append((DIAS[i], FRANJAS[j]))
        r_local = np.random.default_rng(SEMILLA + k)
        for e, (tipo, d, dif) in ESCENARIOS.items():
            ops = franjas[tipo]
            slot = ops[r_local.integers(len(ops))] if ops else (None, None)
            for m in ("Presencial", "Virtual"):
                val = OPCION_VALOR.get(r[c_esc[(e, m)]])
                if val is None:
                    continue
                filas.append(p | {
                    "escenario": e, "dia_semana": slot[0], "franja": slot[1],
                    "cruce_horario": TIPO[tipo][0], "dispuesto_franja": TIPO[tipo][1],
                    "modalidad": m.lower(), "dificultad_tema": dif, "dias_a_evaluacion": d,
                    "respuesta": val, "asistio": int(val >= 3),
                    "tipo_franja": tipo, "plazo_tramo": tramo_plazo(d),
                })
    return pd.DataFrame(perfiles), horarios, pd.DataFrame(filas)


# ============================================================ 2. calibración
class Escalador:
    def __init__(self, df):
        self.mu = {c: df[c].mean() for c in ["promedio_previo", "asistencia_hist", "carga_cursos",
                                             "veces_llevado_curso", "tiempo_traslado",
                                             "uso_recursos_alternativos", "dificultad_tema"]}
        self.sd = {c: df[c].std() or 1.0 for c in self.mu}

    def z(self, df, c):
        v = df[c].fillna(self.mu[c]) if c == "asistencia_hist" else df[c]
        return (v - self.mu[c]) / self.sd[c]


def disenar(df, esc, tau):
    """Matriz de diseño de la ec. (3) con variables estandarizadas."""
    urg = np.exp(-df["dias_a_evaluacion"] / tau)
    presencial = (df["modalidad"] == "presencial").astype(float)
    X = pd.DataFrame({
        "asistencia_hist": esc.z(df, "asistencia_hist"),
        "sin_historial": df["sin_historial"].astype(float),
        "carga_cursos": esc.z(df, "carga_cursos"),
        "veces_llevado_curso": esc.z(df, "veces_llevado_curso"),
        "uso_recursos_alternativos": esc.z(df, "uso_recursos_alternativos"),
        "cruce_horario": df["cruce_horario"].astype(float),
        "dispuesto_franja": df["dispuesto_franja"].astype(float),
        "virtual": 1 - presencial,
        "dificultad_tema": esc.z(df, "dificultad_tema"),
        "urgencia": urg,
        "urgencia_x_promedio_bajo": urg * (-esc.z(df, "promedio_previo")),
        "traslado_x_presencial": esc.z(df, "tiempo_traslado") * presencial,
    })
    return X


def ajustar(X, y, C=1.0):
    lr = LogisticRegression(C=C, max_iter=5000)
    lr.fit(X, y)
    return lr.intercept_[0], pd.Series(lr.coef_[0], index=X.columns)


def calibrar(semilla_cal):
    esc = Escalador(semilla_cal.drop_duplicates("id_estudiante"))
    y = semilla_cal["asistio"].values

    # τ: el que mejor explica la semilla de calibración (log-verosimilitud)
    mejor = None
    for tau in [2, 3, 5, 7, 10, 14]:
        X = disenar(semilla_cal, esc, tau)
        b0, b = ajustar(X, y)
        p = 1 / (1 + np.exp(-(b0 + X.values @ b.values)))
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if mejor is None or ll > mejor[0]:
            mejor = (ll, tau)
    tau = mejor[1]
    X = disenar(semilla_cal, esc, tau)
    b0, beta = ajustar(X, y)

    # Bootstrap por estudiante: intervalos de los βj
    ids = semilla_cal["id_estudiante"].unique()
    boots = []
    for _ in range(300):
        muestra = rng.choice(ids, size=len(ids), replace=True)
        d = pd.concat([semilla_cal[semilla_cal["id_estudiante"] == i] for i in muestra])
        if d["asistio"].nunique() < 2:
            continue
        boots.append(ajustar(disenar(d, esc, tau), d["asistio"].values)[1])
    boots = pd.DataFrame(boots)
    ic = pd.DataFrame({"estimado": beta, "ic90_inf": boots.quantile(0.05), "ic90_sup": boots.quantile(0.95)})

    # Criterio experto: si el signo contradice al experto, el efecto se anula
    ic["usado"] = ic["estimado"]
    ic["ajuste"] = ""
    for v, s in SIGNO_EXPERTO.items():
        if np.sign(ic.loc[v, "estimado"]) not in (0, s):
            ic.loc[v, "usado"] = 0.0
            ic.loc[v, "ajuste"] = "signo opuesto al criterio experto -> 0"

    # Efecto aleatorio por estudiante: σ_u y β0 por simulación, corrigiendo la
    # atenuación de los coeficientes poblacionales (factor de Zeger et al.)
    tasas_obs = semilla_cal.groupby("id_estudiante")["asistio"].mean()
    var_obs, media_obs = tasas_obs.var(), y.mean()
    Xv = X.values
    mejor = None
    for su in np.linspace(0, 2.5, 26):
        f = np.sqrt(1 + 0.346 * su ** 2)
        bcond = ic["usado"].values * f
        for b0c in np.linspace(-3, 3, 61):
            vs, ms = [], []
            for rep in range(20):
                u = rng.normal(0, su, len(ids))
                u_fila = pd.Series(u, index=ids)[semilla_cal["id_estudiante"]].values
                z = b0c + Xv @ bcond + u_fila + rng.normal(0, SIGMA_EPS, len(Xv))
                ysim = rng.random(len(z)) < 1 / (1 + np.exp(-z))
                t = pd.Series(ysim).groupby(semilla_cal["id_estudiante"].values).mean()
                vs.append(t.var()); ms.append(ysim.mean())
            err = (np.mean(vs) - var_obs) ** 2 / var_obs ** 2 + 4 * (np.mean(ms) - media_obs) ** 2
            if mejor is None or err < mejor[0]:
                mejor = (err, su, b0c, f)
    _, sigma_u, beta0, factor = mejor
    ic["condicional"] = ic["usado"] * factor

    return {"escalador": esc, "tau": tau, "beta0": beta0, "sigma_u": sigma_u,
            "coef": ic, "tasa_global": media_obs, "var_entre_estudiantes": var_obs}


# ============================================================ 3. generación
def ajustar_perfiles(perf):
    """Cópula gaussiana con marginales empíricas (rangos normalizados)."""
    cols = ["promedio_previo", "asistencia_hist_imp", "sin_historial", "carga_cursos",
            "veces_llevado_curso", "tiempo_traslado", "uso_recursos_alternativos"]
    p = perf.copy()
    p["asistencia_hist_imp"] = p["asistencia_hist"].fillna(-1)  # -1 marca "sin historial"
    u = p[cols].rank(method="average").values / (len(p) + 1)
    z = stats.norm.ppf(u)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr) + np.eye(len(cols)) * 1e-6
    return {"cols": cols, "corr": corr, "datos": p[cols].values}


def muestrear_perfiles(modelo, n, start_id):
    cols, corr, datos = modelo["cols"], modelo["corr"], modelo["datos"]
    z = rng.multivariate_normal(np.zeros(len(cols)), corr, size=n)
    u = stats.norm.cdf(z)
    out = {}
    for j, c in enumerate(cols):
        v = np.sort(datos[:, j])
        idx = np.clip((u[:, j] * len(v)).astype(int), 0, len(v) - 1)
        out[c] = v[idx]
    df = pd.DataFrame(out)
    # pequeña variación en las continuas para no copiar valores exactos de la semilla
    df["promedio_previo"] = np.clip(df["promedio_previo"] + rng.normal(0, 0.6, n), 0, 20).round(2)
    df["tiempo_traslado"] = np.clip(df["tiempo_traslado"] * rng.lognormal(0, 0.15, n), 5, 180).round()
    sin = df["asistencia_hist_imp"] < 0
    df["sin_historial"] = sin.astype(int)
    df["asistencia_hist"] = np.where(sin, np.nan,
                                     np.clip(df["asistencia_hist_imp"] + rng.normal(0, 0.05, n), 0, 1).round(2))
    df = df.drop(columns="asistencia_hist_imp")
    for c in ["carga_cursos", "veces_llevado_curso", "uso_recursos_alternativos"]:
        df[c] = df[c].astype(int)
    df.insert(0, "id_estudiante", [f"sint_{start_id + i:04d}" for i in range(n)])
    return df


def ajustar_horario(horarios):
    ocu = np.mean([h[0] for h in horarios.values()], axis=0)
    libres = np.sum([1 - h[0] for h in horarios.values()], axis=0)
    aco = np.sum([h[1] for h in horarios.values()], axis=0) / np.maximum(libres, 1)
    return np.clip(ocu, 0.03, 0.97), np.clip(aco, 0.03, 0.97)


def fase_y_dias(n):
    """Escenarios del ciclo (Sec. C): inicio, previo y posterior a evaluaciones."""
    fase = rng.choice(["inicio", "previo_evaluacion", "posterior_evaluacion"], size=n, p=[0.3, 0.4, 0.3])
    d = np.where(fase == "inicio", rng.integers(15, 31, n),
                 np.where(fase == "previo_evaluacion", rng.integers(1, 8, n), rng.integers(8, 22, n)))
    return fase, d


def generar(cal, perf_modelo, p_ocu, p_aco, n_est, sesiones_por_est):
    perf = muestrear_perfiles(perf_modelo, n_est, 1)
    filas = []
    for _, s in perf.iterrows():
        ocu = rng.random((6, 3)) < p_ocu
        aco = (~ocu) & (rng.random((6, 3)) < p_aco)
        u = rng.normal(0, cal["sigma_u"])
        fase, dias = fase_y_dias(sesiones_por_est)
        for k in range(sesiones_por_est):
            i, j = rng.integers(6), rng.integers(3)
            filas.append(s.to_dict() | {
                "dia_semana": DIAS[i], "franja": FRANJAS[j],
                "cruce_horario": int(ocu[i, j]), "dispuesto_franja": int(aco[i, j]),
                "modalidad": rng.choice(["presencial", "virtual"]),
                "dificultad_tema": int(rng.integers(1, 6)),
                "dias_a_evaluacion": int(dias[k]), "fase_ciclo": fase[k], "_u": u,
            })
    df = pd.DataFrame(filas)
    X = disenar(df, cal["escalador"], cal["tau"])
    b = cal["coef"]["condicional"].reindex(X.columns).values
    z = (cal["beta0"] + X.values @ b + df["_u"].values
         + df["franja"].map(EFECTO_FRANJA).values + df["dia_semana"].map(EFECTO_DIA).values
         + rng.normal(0, SIGMA_EPS, len(df)))
    df["p_asistir"] = (1 / (1 + np.exp(-z))).round(4)
    df["asistio"] = (rng.random(len(df)) < df["p_asistir"]).astype(int)  # y ~ Bernoulli(p)
    df["tipo_franja"] = np.select([df["cruce_horario"] == 1, df["dispuesto_franja"] == 1],
                                  ["ocupada", "acomoda"], "libre")
    df["plazo_tramo"] = df["dias_a_evaluacion"].map(tramo_plazo)
    return df.drop(columns="_u")


# ====================================================== 4. combinaciones
def clave(df):
    return (df["tipo_franja"] + "|" + df["plazo_tramo"] + "|" +
            df["dificultad_tema"].astype(str) + "|" + df["modalidad"])


def elegir_reservadas(frac=0.15, escenarios_reales=("E3", "E8")):
    """Combinaciones tipo_franja × plazo × dificultad × modalidad (150) que NO
    aparecerán en entrenamiento. Incluye dos escenarios reales del formulario,
    para que la prueba real también tenga combinaciones no vistas."""
    todas = [f"{t}|{p}|{d}|{m}" for t in ("ocupada", "libre", "acomoda")
             for p in ("manana", "3a5dias", "1semana", "2semanas", "1mes")
             for d in range(1, 6) for m in ("presencial", "virtual")]
    res = set()
    for e in escenarios_reales:
        t, d, dif = ESCENARIOS[e]
        for m in ("presencial", "virtual"):
            res.add(f"{t}|{tramo_plazo(d)}|{dif}|{m}")
    resto = [c for c in todas if c not in res]
    rng.shuffle(resto)
    objetivo = int(round(frac * len(todas)))
    for c in resto:
        if len(res) >= objetivo:
            break
        res.add(c)
    return sorted(res), todas


# ================================================================== main
COLS_MODELO = ["promedio_previo", "asistencia_hist", "sin_historial", "carga_cursos",
               "veces_llevado_curso", "tiempo_traslado", "uso_recursos_alternativos",
               "dia_semana", "franja", "cruce_horario", "dispuesto_franja",
               "modalidad", "dificultad_tema", "dias_a_evaluacion"]


def main(ruta_csv, salida, n_est=600, sesiones=20):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    # 1. semilla y división por estudiante (estratificada por tasa de asistencia)
    perf, horarios, semilla = leer_semilla(ruta_csv)
    tasas = semilla.groupby("id_estudiante")["asistio"].mean().sort_values()
    ids = list(tasas.index)
    prueba_ids = set(ids[1::3])  # 1 de cada 3, repartidos en todo el rango de tasas
    cal_ids = set(ids) - prueba_ids
    semilla_cal = semilla[semilla["id_estudiante"].isin(cal_ids)].copy()
    semilla_test = semilla[semilla["id_estudiante"].isin(prueba_ids)].copy()

    # 2. calibración (solo con la semilla de calibración)
    cal = calibrar(semilla_cal)
    perf_modelo = ajustar_perfiles(perf[perf["id_estudiante"].isin(cal_ids)])
    p_ocu, p_aco = ajustar_horario({k: v for k, v in horarios.items() if k in cal_ids})

    # 3. generación
    sint = generar(cal, perf_modelo, p_ocu, p_aco, n_est, sesiones)

    # 4. particiones
    reservadas, todas = elegir_reservadas()
    sint["combinacion"] = clave(sint)
    sint["combo_reservada"] = sint["combinacion"].isin(reservadas).astype(int)
    semilla_test["combinacion"] = clave(semilla_test)
    semilla_test["combo_reservada"] = semilla_test["combinacion"].isin(reservadas).astype(int)

    est = np.array(sint["id_estudiante"].unique(), dtype=object)
    rng.shuffle(est)
    n_test = int(0.2 * len(est))
    test_est = set(est[:n_test])
    train_est = list(est[n_test:])
    fold = {e: k % 5 for k, e in enumerate(train_est)}  # 5 pliegues por estudiante

    tr = sint[sint["id_estudiante"].isin(train_est) & (sint["combo_reservada"] == 0)].copy()
    tr["fold"] = tr["id_estudiante"].map(fold)
    te_est = sint[sint["id_estudiante"].isin(test_est) & (sint["combo_reservada"] == 0)].copy()
    te_combo = sint[sint["id_estudiante"].isin(test_est) & (sint["combo_reservada"] == 1)].copy()
    # Las filas de estudiantes de entrenamiento con combinaciones reservadas se descartan.

    base = ["id_estudiante"] + COLS_MODELO + ["asistio"]
    aux = ["tipo_franja", "plazo_tramo", "combinacion", "combo_reservada"]
    tr[base + ["fold"] + aux + ["fase_ciclo", "p_asistir"]].to_csv(salida / "sintetico_train.csv", index=False)
    te_est[base + aux + ["fase_ciclo", "p_asistir"]].to_csv(salida / "sintetico_test_estudiantes.csv", index=False)
    te_combo[base + aux + ["fase_ciclo", "p_asistir"]].to_csv(salida / "sintetico_test_combinaciones.csv", index=False)
    semilla_test[base + ["escenario", "respuesta"] + aux].to_csv(salida / "semilla_real_prueba.csv", index=False)
    semilla_cal[base + ["escenario", "respuesta", "tipo_franja", "plazo_tramo"]].to_csv(
        salida / "semilla_real_calibracion.csv", index=False)
    pd.DataFrame({"combinacion": reservadas}).to_csv(salida / "combinaciones_reservadas.csv", index=False)

    # 5. validación del generador: tasas por escenario con el diseño del formulario
    val = validar_por_escenario(cal, perf_modelo, p_ocu, p_aco, semilla_cal, semilla_test)
    val.to_csv(salida / "validacion_por_escenario.csv", index=False)

    resumen = {
        "estudiantes_reales": len(ids), "calibracion": len(cal_ids), "prueba_real": len(prueba_ids),
        "tau": cal["tau"], "beta0": round(cal["beta0"], 3), "sigma_u": round(cal["sigma_u"], 3),
        "sigma_eps": SIGMA_EPS,
        "tasa_semilla_cal": round(cal["tasa_global"], 3),
        "tasa_semilla_test": round(semilla_test["asistio"].mean(), 3),
        "tasa_sintetica": round(sint["asistio"].mean(), 3),
        "filas": {"sintetico_train": len(tr), "sintetico_test_estudiantes": len(te_est),
                  "sintetico_test_combinaciones": len(te_combo), "semilla_real_prueba": len(semilla_test),
                  "semilla_real_prueba_combos_no_vistas": int(semilla_test["combo_reservada"].sum())},
        "combinaciones_reservadas": f"{len(reservadas)} de {len(todas)}",
    }
    cal["coef"].round(3).to_csv(salida / "coeficientes_calibrados.csv")
    with open(salida / "resumen_calibracion.json", "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    return resumen, cal, val


def validar_por_escenario(cal, perf_modelo, p_ocu, p_aco, semilla_cal, semilla_test, n=2000):
    """Simula estudiantes sintéticos respondiendo los 10 escenarios del formulario."""
    perf = muestrear_perfiles(perf_modelo, n, 90000)
    filas = []
    for _, s in perf.iterrows():
        u = rng.normal(0, cal["sigma_u"])
        for e, (tipo, d, dif) in ESCENARIOS.items():
            for m in ("presencial", "virtual"):
                filas.append(s.to_dict() | {"escenario": e, "cruce_horario": TIPO[tipo][0],
                                            "dispuesto_franja": TIPO[tipo][1], "modalidad": m,
                                            "dificultad_tema": dif, "dias_a_evaluacion": d, "_u": u})
    df = pd.DataFrame(filas)
    X = disenar(df, cal["escalador"], cal["tau"])
    z = (cal["beta0"] + X.values @ cal["coef"]["condicional"].reindex(X.columns).values
         + df["_u"].values + rng.normal(0, SIGMA_EPS, len(df)))
    df["asistio"] = (rng.random(len(df)) < 1 / (1 + np.exp(-z))).astype(int)
    out = pd.DataFrame({
        "tasa_sintetica": df.groupby(["escenario", "modalidad"])["asistio"].mean(),
        "tasa_real_calibracion": semilla_cal.groupby(["escenario", "modalidad"])["asistio"].mean(),
        "tasa_real_prueba": semilla_test.groupby(["escenario", "modalidad"])["asistio"].mean(),
    }).round(3).reset_index()
    out["orden"] = out["escenario"].str[1:].astype(int)
    return out.sort_values(["orden", "modalidad"]).drop(columns="orden")


if __name__ == "__main__":
    r, cal, val = main(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(cal["coef"].round(3).to_string())
    print(val.to_string(index=False))
