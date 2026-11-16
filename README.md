# Modelo predictivo de asistencia a asesorías académicas mediante algoritmos de clasificación supervisada

**Curso:** Inteligencia Artificial (1INF24), Pontificia Universidad Católica del Perú
**Grupo 6:** Sebastián Mathew Camborda Salazar, Sebastián Rodrigo Huerta Felipa, William He Wang, Emmet Arther Schafft

## Resumen

Las asesorías académicas suelen programarse por intuición o con encuestas de disponibilidad, que indican cuándo un estudiante *puede* asistir, pero no cuándo es *probable* que lo haga. Este proyecto formula el problema como una clasificación binaria: dado el perfil de un estudiante y una sesión programada, predecir si asistirá (1) o no (0). Sumando las probabilidades predichas de los estudiantes de un curso se estima la asistencia esperada de cada horario candidato, lo que permite elegir el horario con mayor asistencia esperada.

Se comparan cinco algoritmos de clasificación supervisada: K-Nearest Neighbors (KNN) y el árbol de decisión, dos modelos simples (el árbol, además, se lee como reglas lógicas); Random Forest y Gradient Boosting, dos ensambles de árboles; y Naive Bayes. Un clasificador trivial que siempre predice la clase mayoritaria sirve como línea base. Los modelos se entrenan con datos sintéticos calibrados a partir de respuestas reales de un formulario, y se evalúan también en un conjunto de estudiantes reales que el generador nunca vio.

## 1. Pregunta de investigación e hipótesis

**Pregunta.** ¿Es posible predecir, a partir de variables académicas y temporales, si un estudiante de Ingeniería Informática asistirá a una sesión de asesoría programada? ¿Qué algoritmo de clasificación ofrece el mejor desempeño y cuánto se mantiene ese desempeño con estudiantes reales?

- **H1.** El árbol de decisión alcanzará una balanced accuracy igual o superior a la de KNN.
- **H2.** Los ensambles (Random Forest y Gradient Boosting) obtendrán una balanced accuracy mayor que la de los modelos individuales (KNN, árbol de decisión y Naive Bayes), porque al combinar muchos árboles corrigen los errores de un árbol aislado: Random Forest reduce su varianza y Gradient Boosting, su sesgo.
- **H3.** El desempeño del modelo seleccionado se mantendrá con estudiantes y combinaciones de sesión simulados que no vio, pero bajará con los estudiantes reales, porque el generador sintético no reproduce todo el comportamiento real de la semilla.

## 2. Objetivos

- **General.** Desarrollar y comparar cinco modelos de clasificación supervisada (KNN, árbol de decisión, Random Forest, Gradient Boosting y Naive Bayes) que predigan la asistencia de un estudiante a una sesión de asesoría programada, y seleccionar el de mejor desempeño.
- **OE1.** Construir un conjunto de datos que combine datos reales semilla, obtenidos mediante un formulario, con datos sintéticos probabilísticos, junto con un pipeline de preprocesamiento.
- **OE2.** Entrenar los modelos y ajustar sus hiperparámetros mediante búsqueda en malla con validación cruzada agrupada por estudiante, para controlar el sobreajuste y la fuga de información.
- **OE3.** Comparar los modelos con la balanced accuracy, como métrica principal, y con Accuracy, Precision, Recall y F1-Score, frente a una línea base trivial, y seleccionar el modelo final antes de usar los conjuntos de prueba.
- **OE4.** Evaluar el modelo seleccionado con estudiantes nuevos, combinaciones de sesión no vistas y estudiantes reales, y analizar sus errores por clase y la asistencia total que estima.

## 3. Datos

Cada fila corresponde a un par (estudiante, sesión) y la variable objetivo es `asistio` (1 = asistió, 0 = no asistió). Las particiones vienen divididas **por estudiante**: ningún estudiante aparece en dos particiones, de modo que el modelo siempre se evalúa con estudiantes que no vio.

| Archivo | Filas | Uso |
|---|---|---|
| `dataset/sintetico_train.csv` | 7 990 | Entrenamiento y validación cruzada. 480 estudiantes simulados; la columna `fold` (0 a 4) define 5 pliegues agrupados por estudiante. |
| `dataset/sintetico_test_estudiantes.csv` | 2 030 | Prueba 1: 120 estudiantes simulados nuevos, con combinaciones de sesión vistas en entrenamiento. |
| `dataset/sintetico_test_combinaciones.csv` | 370 | Prueba 2: los mismos 120 estudiantes, solo con combinaciones de sesión que no existen en entrenamiento. |
| `dataset/semilla_real_prueba.csv` | 200 | Prueba final con datos reales: 10 estudiantes del formulario que el generador nunca vio. |
| `dataset/combinaciones_reservadas.csv` | 22 | Combinaciones de sesión excluidas del entrenamiento (referencia). |

**Variables de entrada (14).** `promedio_previo`, `asistencia_hist`, `sin_historial`, `carga_cursos`, `veces_llevado_curso`, `tiempo_traslado`, `uso_recursos_alternativos`, `dia_semana`, `franja`, `cruce_horario`, `dispuesto_franja`, `modalidad`, `dificultad_tema` y `dias_a_evaluacion`. Las demás columnas (`id_estudiante`, `fold`, `p_asistir`, `combinacion`, etc.) son auxiliares y no se usan como entrada.

**Distribución de clases.** Las clases están moderadamente desbalanceadas: cerca del 61 % de las filas sintéticas y el 65 % de la semilla real corresponden a "asiste".

**Generación sintética.** Las etiquetas sintéticas se generaron con un modelo logístico calibrado con 20 estudiantes reales (perfiles mediante cópula gaussiana, efecto aleatorio por estudiante y ruido por sesión). El generador y los archivos de calibración se encuentran en el historial del repositorio (commit `1bb7d80`).

## 4. Metodología

El flujo completo está en `Modelo_Prediccion_Asesorias.ipynb`.

1. **Separación de datos.** Los conjuntos de prueba se mantienen aparte desde el inicio y se usan una sola vez, al final.
2. **Análisis exploratorio** sobre el conjunto de entrenamiento: dimensiones, nulos, tipos, distribución del objetivo y correlaciones.
3. **Preprocesamiento dentro de un Pipeline** de scikit-learn, para evitar fuga de información: se ajusta solo con los datos de entrenamiento de cada pliegue.
   - Imputación con la mediana en variables numéricas y con la moda en categóricas. Los únicos nulos numéricos son los de `asistencia_hist`, que corresponden a estudiantes sin historial de asesorías (marcados por `sin_historial`).
   - One-hot encoding de `dia_semana`, `franja` y `modalidad`.
   - Normalización min-max, aplicada por igual a todos los modelos.
4. **Línea base.** Un clasificador trivial (`DummyClassifier`) que siempre predice "asiste". Por el desbalance de clases obtiene un F1 cercano a 0.76 sin aprender nada; un modelo útil debe superarlo con claridad, sobre todo en balanced accuracy, donde la línea base obtiene 0.5.
5. **Ajuste de hiperparámetros** con `GridSearchCV` sobre los 5 pliegues agrupados por estudiante (`PredefinedSplit`):

   | Modelo | Tipo | Hiperparámetros explorados |
   |---|---|---|
   | Árbol de decisión (DT) | Simple | `max_depth` 2–12, `min_samples_leaf` {1, 5, 10, 20, 50}, `ccp_alpha` {0, 0.0005, 0.001, 0.005, 0.01} |
   | KNN | Simple | K impar de 1 a 25, distancia euclidiana o Manhattan, ponderación uniforme o por distancia |
   | Random Forest (RT) | Ensamble | árboles {100, 300}, profundidad {5, 10, 15, sin límite}, hoja mínima {1, 5, 20, 50} |
   | Gradient Boosting (GB) | Ensamble | árboles {100, 200}, tasa de aprendizaje {0.05, 0.1}, profundidad {2, 3} |
   | Naive Bayes (NB) | Probabilístico | `var_smoothing` {1e-9, 1e-7, 1e-5, 1e-3} |

   En total se prueban 371 combinaciones (1 855 entrenamientos), todas con los mismos pliegues.

   La regresión logística no se incluye porque las etiquetas sintéticas provienen de un modelo logístico, lo que le daría una ventaja artificial.
6. **Métrica principal: balanced accuracy** (promedio del recall de cada clase). Se prefiere sobre el F1-Score porque castiga por igual los errores en ambas clases y, con clases desbalanceadas, no premia sobreestimar la asistencia, lo que es clave para estimar cuántos estudiantes asistirán a cada horario. Accuracy, Precision, Recall y F1 se reportan como métricas complementarias.
7. **Selección del modelo** por la mayor balanced accuracy media en validación cruzada, antes de mirar los conjuntos de prueba.
8. **Evaluación final.** El modelo seleccionado se reentrena con todo el conjunto de entrenamiento y se evalúa en los tres conjuntos de prueba (balanced accuracy, matriz de confusión y reporte de clasificación). Los demás modelos se muestran solo como comparación.

## 5. Resultados

### 5.1 Validación cruzada (5 pliegues agrupados por estudiante)

| Modelo | Balanced accuracy | F1 | Accuracy |
|---|---|---|---|
| Gradient Boosting | **0.752 ± 0.017** | 0.815 ± 0.014 | 0.771 ± 0.012 |
| Random Forest | 0.745 ± 0.017 | 0.814 ± 0.014 | 0.766 ± 0.012 |
| Árbol de decisión | 0.734 ± 0.011 | 0.796 ± 0.016 | 0.750 ± 0.009 |
| Naive Bayes | 0.727 ± 0.021 | 0.776 ± 0.019 | 0.735 ± 0.019 |
| KNN | 0.708 ± 0.013 | 0.788 ± 0.017 | 0.732 ± 0.014 |
| Línea base | 0.500 ± 0.000 | 0.756 ± 0.032 | 0.608 ± 0.041 |

Mejores hiperparámetros: árbol con profundidad 8, `min_samples_leaf` = 20 y `ccp_alpha` = 0.0005; KNN con K = 25, distancia Manhattan y ponderación por distancia; Random Forest con 100 árboles, sin límite de profundidad y hoja mínima 1; Gradient Boosting con 200 árboles de profundidad 2 y tasa de aprendizaje 0.1; Naive Bayes con `var_smoothing` = 1e-9.

### 5.2 Conjuntos de prueba (balanced accuracy)

| Modelo | Estudiantes nuevos | Combinaciones no vistas | Semilla real | Asistentes predichos en la semilla real (reales: 130) |
|---|---|---|---|---|
| **Gradient Boosting (seleccionado)** | 0.756 | 0.763 | 0.671 | 156 |
| Random Forest | 0.745 | 0.725 | 0.686 | 154 |
| Árbol de decisión | 0.731 | 0.732 | 0.649 | 156 |
| Naive Bayes | 0.722 | 0.731 | 0.676 | 143 |
| KNN | 0.710 | 0.733 | 0.692 | 147 |
| Línea base | 0.500 | 0.500 | 0.500 | 200 |

### 5.3 Interpretación

- Todos los modelos superan claramente a la línea base. **Modelo seleccionado:** Gradient Boosting, con la mayor balanced accuracy en validación cruzada (0.752), seguido de cerca por Random Forest (0.745).
- **H1 (árbol ≥ KNN):** se cumple en validación cruzada (0.734 frente a 0.708) y con estudiantes nuevos (0.731 frente a 0.710). En combinaciones no vistas ambos empatan (0.732 y 0.733) y en la semilla real el orden se invierte (0.649 frente a 0.692), aunque con solo 200 filas de 10 estudiantes la diferencia no es concluyente.
- **H2 (ensambles > modelos individuales):** se cumple en validación cruzada y con estudiantes nuevos (0.756 y 0.745 frente a un máximo de 0.731). Con combinaciones no vistas Random Forest (0.725) queda por debajo de los modelos individuales, y en la semilla real el mejor modelo es KNN (0.692).
- **H3 (se mantiene en datos sintéticos, baja en datos reales):** se cumple. El modelo seleccionado mantiene el nivel de la validación cruzada con estudiantes nuevos y combinaciones no vistas (0.756 y 0.763 frente a 0.752), lo que indica que generaliza y no memoriza combinaciones, pero baja a 0.671 en la semilla real, como todos los modelos.
- **Errores por clase y asistencia estimada:** en la semilla real, Gradient Boosting identifica 117 de los 130 casos de asistencia, pero solo 31 de los 70 de no asistencia (recall de 0.44 en esa clase). Con umbral de 0.5, todos los modelos sobreestiman la asistencia total entre 10 % y 20 %.

## 6. Limitaciones

- Las etiquetas sintéticas provienen de un modelo definido por el equipo, por lo que el desempeño en datos sintéticos no equivale al desempeño en el mundo real. La semilla real es el criterio más confiable, pero tiene solo 10 estudiantes.
- La asistencia de la semilla real es declarada en un formulario, no observada.
- Naive Bayes supone variables normales e independientes dentro de cada clase, lo que no se cumple del todo en estos datos.
- Con umbral de 0.5, los modelos predicen más asistentes que los reales; para estimar la asistencia por horario conviene sumar las probabilidades predichas en lugar de contar las predicciones positivas.
- Los mejores hiperparámetros de algunos modelos quedaron en el borde de la grilla explorada (por ejemplo, K = 25), por lo que una grilla más amplia podría mejorar ligeramente los resultados.
- No se analizaron la importancia de las variables, la sensibilidad a los hiperparámetros ni las reglas del árbol de decisión; quedan como trabajo futuro.

## 7. Estructura del repositorio

```
├── Modelo_Prediccion_Asesorias.ipynb   # análisis completo: preprocesamiento, ajuste, selección y evaluación
├── dataset/
│   ├── LEEME.txt                       # descripción de archivos y columnas del dataset
│   ├── sintetico_train.csv
│   ├── sintetico_test_estudiantes.csv
│   ├── sintetico_test_combinaciones.csv
│   ├── semilla_real_prueba.csv
│   └── combinaciones_reservadas.csv
└── README.md
```

## 8. Cómo reproducir

Requisitos (versiones con las que se ejecutó el notebook): Python 3.14, pandas 3.0, numpy 2.4, scikit-learn 1.9, matplotlib 3.11 y seaborn 0.13.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

```bash
jupyter notebook Modelo_Prediccion_Asesorias.ipynb
```

El notebook debe ejecutarse desde la raíz del repositorio, porque lee los datos con rutas relativas (`dataset/...`). Todos los modelos con componente aleatorio usan `random_state=42`, por lo que los resultados son reproducibles. La ejecución completa toma alrededor de 2 minutos.

## Referencias

1. C. Romero and S. Ventura, "Educational data mining: A review of the state of the art," *IEEE Trans. Syst., Man, Cybern. C, Appl. Rev.*, vol. 40, no. 6, pp. 601–618, Nov. 2010.
2. I. Muzaferija, Z. Mašetić, S. Jukić, and D. Kečo, "Student attendance pattern detection and prediction," *J. Natural Sciences and Engineering (JONSAE)*, 2021, doi: 10.14706/JONSAE2021313.
3. G. Diaz Ramos et al., "Stable and privacy-preserving synthetic educational data with empirical marginals: A copula-based approach," in *Proc. 19th Int. Conf. Educational Data Mining (EDM 2026)*, Seoul, 2026. arXiv:2604.04195.
4. T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Inf. Theory*, vol. 13, no. 1, pp. 21–27, Jan. 1967.
5. L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*. Belmont, CA, USA: Wadsworth, 1984.
6. R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in *Proc. 14th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 1995, pp. 1137–1143.
7. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
8. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
9. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
10. K. H. Brodersen, C. S. Ong, K. E. Stephan, and J. M. Buhmann, "The balanced accuracy and its posterior distribution," in *Proc. 20th Int. Conf. Pattern Recognition (ICPR)*, 2010, pp. 3121–3124.
