"""
Verificación de las particiones y prueba rápida de KNN y árbol de decisión.
Uso: python verificar_particiones.py carpeta_salida

Sirve también como plantilla del pipeline de la Sec. D del informe:
imputación + one-hot + escalado (solo KNN), ajuste solo con entrenamiento,
GridSearchCV con pliegues agrupados por estudiante (columna "fold").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

d = Path(sys.argv[1])
tr = pd.read_csv(d / "sintetico_train.csv")
te_est = pd.read_csv(d / "sintetico_test_estudiantes.csv")
te_combo = pd.read_csv(d / "sintetico_test_combinaciones.csv")
real = pd.read_csv(d / "semilla_real_prueba.csv")
cal = pd.read_csv(d / "semilla_real_calibracion.csv")
reservadas = set(pd.read_csv(d / "combinaciones_reservadas.csv")["combinacion"])

# ---------------- 1. comprobaciones de fuga ----------------
def ids(df): return set(df["id_estudiante"])
checks = {
    "train y test_estudiantes no comparten estudiantes": not (ids(tr) & ids(te_est)),
    "train y test_combinaciones no comparten estudiantes": not (ids(tr) & ids(te_combo)),
    "semilla de prueba y de calibración no comparten estudiantes": not (ids(real) & ids(cal)),
    "ninguna combinación reservada aparece en train": not (set(tr["combinacion"]) & reservadas),
    "test_combinaciones solo tiene combinaciones reservadas": set(te_combo["combinacion"]) <= reservadas,
    "cada fold agrupa estudiantes completos": tr.groupby("id_estudiante")["fold"].nunique().max() == 1,
}
for c in ["tipo_franja", "plazo_tramo", "dificultad_tema", "modalidad", "dia_semana", "franja"]:
    checks[f"todos los niveles de {c} aparecen en train"] = set(te_combo[c]) <= set(tr[c])
print("Comprobaciones de fuga de información")
for k, v in checks.items():
    print(f"  [{'OK' if v else 'FALLA'}] {k}")

# ---------------- 2. pipeline y modelos ----------------
NUM = ["promedio_previo", "asistencia_hist", "sin_historial", "carga_cursos", "veces_llevado_curso",
       "tiempo_traslado", "uso_recursos_alternativos", "cruce_horario", "dispuesto_franja",
       "dificultad_tema", "dias_a_evaluacion"]
CAT = ["dia_semana", "franja", "modalidad"]
X_COLS = NUM + CAT  # sin id_estudiante, escenario ni combinacion: serían atajos para memorizar

def prep(escalar):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("esc", StandardScaler())] if escalar else [])
    cat = [("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num), NUM), ("cat", Pipeline(cat), CAT)])

cv = PredefinedSplit(tr["fold"].values)  # pliegues agrupados por estudiante
modelos = {
    "Línea base (siempre la clase mayoritaria)": GridSearchCV(
        Pipeline([("prep", prep(False)), ("m", DummyClassifier(strategy="most_frequent"))]),
        {}, scoring="f1", cv=cv),
    "Árbol": GridSearchCV(Pipeline([("prep", prep(False)), ("m", DecisionTreeClassifier(random_state=2026))]),
                          {"m__max_depth": [2, 4, 6, 8, 10, 12], "m__min_samples_leaf": [1, 10, 30]},
                          scoring="f1", cv=cv, n_jobs=-1),
    "KNN": GridSearchCV(Pipeline([("prep", prep(True)), ("m", KNeighborsClassifier())]),
                        {"m__n_neighbors": list(range(1, 26, 2)), "m__weights": ["uniform", "distance"]},
                        scoring="f1", cv=cv, n_jobs=-1),
}
pruebas = {"Sintético: estudiantes nuevos": te_est,
           "Sintético: combinaciones no vistas": te_combo,
           "Real: semilla de prueba (todas)": real,
           "Real: semilla de prueba, combinaciones no vistas": real[real["combo_reservada"] == 1],
           "Real: semilla de prueba, combinaciones vistas": real[real["combo_reservada"] == 0]}

print(f"\nTamaños: train={len(tr)}  " + "  ".join(f"{k}={len(v)}" for k, v in pruebas.items()))
filas = []
for nombre, gs in modelos.items():
    gs.fit(tr[X_COLS], tr["asistio"])
    for pn, df in pruebas.items():
        yp = gs.predict(df[X_COLS])
        pp = gs.predict_proba(df[X_COLS])[:, 1]
        auc = roc_auc_score(df["asistio"], pp) if df["asistio"].nunique() > 1 else np.nan
        filas.append({"modelo": nombre, "conjunto": pn, "auc": auc,
                      "accuracy": accuracy_score(df["asistio"], yp),
                      "precision": precision_score(df["asistio"], yp, zero_division=0),
                      "recall": recall_score(df["asistio"], yp, zero_division=0),
                      "f1": f1_score(df["asistio"], yp, zero_division=0)})
    print(f"{nombre}: mejores hiperparámetros {gs.best_params_}, F1 en validación cruzada {gs.best_score_:.3f}")
res = pd.DataFrame(filas).round(3)
print(res.to_string(index=False))
res.to_csv(d / "resultados_prueba_rapida.csv", index=False)
