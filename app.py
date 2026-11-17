from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
SEMILLA = 2606

COLUMNAS = [
    "age", "sex", "chest_pain", "blood_pressure", "serum_cholestoral",
    "fasting_blood_sugar", "electrocardiographic", "max_heart_rate",
    "induced_angina", "ST_depression", "slope", "no_of_vessels", "thal",
    "diagnosis"
]
FEATURES = COLUMNAS[:-1]
NOMBRES = [
    "Random Forest 1", "Random Forest 2",
    "Regresión logística", "Red neuronal (MLP)"
]
METRICAS = ["Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"]

st.set_page_config(
    page_title="Predicción de enfermedad cardiovascular",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def cargar_datos():
    ruta = BASE / "processed.cleveland.data"
    if not ruta.exists():
        raise FileNotFoundError("No se encontró processed.cleveland.data.")
    df = pd.read_csv(ruta, names=COLUMNAS, header=None, na_values="?")
    # Se replica el procedimiento del notebook proporcionado.
    df["no_of_vessels"] = df["no_of_vessels"].fillna(df["no_of_vessels"].mode()[0])
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
    df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x > 0 else 0).astype(int)
    return df

def construir_mlp():
    # Importación diferida: evita cargar TensorFlow al abrir el dashboard.
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(SEMILLA)
    red = keras.Sequential([
        layers.Input(shape=(13,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp_cardiaco")
    red.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return red

def obtener_metricas(y_real, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, pred, labels=[0, 1]).ravel()
    return {
        "Exactitud": accuracy_score(y_real, pred),
        "Precisión": precision_score(y_real, pred, zero_division=0),
        "Recall": recall_score(y_real, pred, zero_division=0),
        "Especificidad": tn / (tn + fp) if (tn + fp) else 0,
        "F1": f1_score(y_real, pred, zero_division=0),
        "AUC": roc_auc_score(y_real, prob),
    }

@st.cache_resource(show_spinner="Preparando los cuatro algoritmos para realizar predicciones...")
def cargar_y_entrenar_modelos():
    df = cargar_datos()
    X = df[FEATURES]
    y = df["diagnosis"]

    # Se conserva exactamente la división descrita en el notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEMILLA
    )

    scaler_path = BASE / "scaler.joblib"
    rf2_path = BASE / "modelo_final.joblib"
    if not scaler_path.exists() or not rf2_path.exists():
        raise FileNotFoundError(
            "Faltan scaler.joblib o modelo_final.joblib. "
            "Deben estar en la misma carpeta que app.py."
        )

    scaler = joblib.load(scaler_path)
    rf2 = joblib.load(rf2_path)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RF1 y Regresión logística: parámetros del notebook.
    rf1 = RandomForestClassifier(n_estimators=100, random_state=SEMILLA)
    rf1.fit(X_train_scaled, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=SEMILLA)
    lr.fit(X_train_scaled, y_train)

    # MLP: arquitectura, optimizador, épocas, batch y early stopping del notebook.
    mlp = construir_mlp()
    from tensorflow import keras
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    mlp.fit(
        X_train_scaled, y_train.to_numpy(),
        validation_split=0.2, epochs=300, batch_size=16,
        callbacks=[early_stop], verbose=0
    )

    modelos = {
        "Random Forest 1": rf1,
        "Random Forest 2": rf2,
        "Regresión logística": lr,
        "Red neuronal (MLP)": mlp,
    }
    return df, scaler, modelos, X_train_scaled, X_test_scaled, y_train, y_test

def probabilidades(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.predict(X, verbose=0).reshape(-1)

def probabilidad_individual(modelo, X_scaled):
    if hasattr(modelo, "predict_proba"):
        return float(modelo.predict_proba(X_scaled)[0, 1])
    return float(modelo.predict(X_scaled, verbose=0).reshape(-1)[0])

def formato_clase(p):
    return "ENFERMEDAD ESTIMADA (1)" if p >= 0.5 else "SIN ENFERMEDAD ESTIMADA (0)"

# Cargar solo información ligera al iniciar. Los modelos se preparan cuando el usuario
# solicita una predicción; así el dashboard puede mostrar su interfaz sin entrenar MLP.
try:
    df = cargar_datos()
    with open(BASE / "info_modelo.json", "r", encoding="utf-8") as f:
        info_modelo = json.load(f)
except Exception as e:
    st.error(f"No se pudieron cargar los datos o la información del modelo: {e}")
    st.stop()

with st.sidebar:
    st.header("Modelos disponibles")
    modelo_principal = st.selectbox("Modelo destacado", NOMBRES, index=1)
    metricas_rf2 = info_modelo.get("metricas_prueba", {})
    if modelo_principal == "Random Forest 2" and metricas_rf2:
        st.metric("Exactitud en prueba", f"{metricas_rf2.get('Exactitud', 0):.1%}")
        st.metric("Recall en prueba", f"{metricas_rf2.get('Recall', 0):.1%}")
        st.metric("AUC en prueba", f"{metricas_rf2.get('AUC', 0):.3f}")
    else:
        st.caption("Las métricas experimentales están disponibles en la pestaña Comparativa.")
    st.caption("Los cuatro algoritmos se ejecutan cuando solicitas una predicción.")
    st.divider()
    st.caption("Datos: 303 pacientes · 13 variables predictoras")

tab_pred, tab_comp, tab_datos = st.tabs(
    ["❤️ Predicción", "📊 Comparativa de modelos", "🗂️ Datos"]
)

# ------------------- PREDICCIÓN -------------------
with tab_pred:
    st.subheader("Datos clínicos del paciente")
    st.write("Completa las variables clínicas. Los cuatro algoritmos devolverán su propia probabilidad.")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Edad (años)", min_value=20, max_value=90, value=55)
        sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: {0: "Mujer", 1: "Hombre"}[x])
        chest_pain = st.selectbox(
            "Tipo de dolor de pecho", [1, 2, 3, 4],
            format_func=lambda x: {
                1: "1 - Angina típica", 2: "2 - Angina atípica",
                3: "3 - Dolor no anginoso", 4: "4 - Asintomático"
            }[x]
        )
        blood_pressure = st.number_input(
            "Presión arterial en reposo (mmHg)", min_value=80, max_value=220, value=130
        )
        serum_cholestoral = st.number_input(
            "Colesterol (mg/dl)", min_value=100, max_value=600, value=245
        )

    with c2:
        fasting_blood_sugar = st.selectbox(
            "Azúcar en ayunas > 120 mg/dl", [0, 1],
            format_func=lambda x: {0: "No", 1: "Sí"}[x]
        )
        electrocardiographic = st.selectbox(
            "Electrocardiograma en reposo", [0, 1, 2],
            format_func=lambda x: {
                0: "0 - Normal", 1: "1 - Anomalía ST-T", 2: "2 - Hipertrofia"
            }[x]
        )
        max_heart_rate = st.number_input(
            "Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150
        )
        induced_angina = st.selectbox(
            "Angina inducida por ejercicio", [0, 1],
            format_func=lambda x: {0: "No", 1: "Sí"}[x]
        )

    with c3:
        ST_depression = st.number_input(
            "Depresión ST por ejercicio", min_value=0.0, max_value=7.0,
            value=1.0, step=0.1
        )
        slope = st.selectbox(
            "Pendiente del segmento ST", [1, 2, 3],
            format_func=lambda x: {
                1: "1 - Ascendente", 2: "2 - Plana", 3: "3 - Descendente"
            }[x]
        )
        no_of_vessels = st.selectbox("Vasos principales coloreados (0-3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thal", [3, 6, 7],
            format_func=lambda x: {
                3: "3 - Normal", 6: "6 - Defecto fijo", 7: "7 - Defecto reversible"
            }[x]
        )

    with st.form("formulario_prediccion"):
        submitted = st.form_submit_button(
            "Evaluar con los cuatro algoritmos", type="primary", use_container_width=True
        )

    if submitted:
        valores = [
            age, sex, chest_pain, blood_pressure, serum_cholestoral,
            fasting_blood_sugar, electrocardiographic, max_heart_rate,
            induced_angina, ST_depression, slope, no_of_vessels, thal
        ]
        fila = pd.DataFrame([valores], columns=FEATURES)
        try:
            with st.spinner("Preparando los cuatro modelos. La primera ejecución puede tardar por el entrenamiento de la red MLP..."):
                _, scaler, modelos, _, _, _, _ = cargar_y_entrenar_modelos()
            fila_scaled = scaler.transform(fila)
        except Exception as e:
            st.error(f"No fue posible preparar los modelos para la predicción: {e}")
            st.stop()

        resultados = []
        for nombre in NOMBRES:
            p = probabilidad_individual(modelos[nombre], fila_scaled)
            clase = int(p >= 0.5)
            resultados.append({
                "Algoritmo": nombre,
                "Probabilidad de clase 1": p,
                "Predicción": clase,
                "Resultado": formato_clase(p)
            })
        resultados_df = pd.DataFrame(resultados)

        st.divider()
        st.subheader("Resultados individuales")
        cards = st.columns(4)
        for col, row in zip(cards, resultados):
            with col:
                st.metric(row["Algoritmo"], f"{row['Probabilidad de clase 1']:.1%}")
                st.progress(float(row["Probabilidad de clase 1"]))
                if row["Predicción"] == 1:
                    st.error(row["Resultado"])
                else:
                    st.success(row["Resultado"])

        st.markdown("#### Tabla de resultados de los cuatro métodos")
        tabla_salida = resultados_df.copy()
        tabla_salida["Probabilidad de clase 1"] = tabla_salida["Probabilidad de clase 1"].map(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(tabla_salida, use_container_width=True, hide_index=True)

        p_destacado = next(
            r["Probabilidad de clase 1"] for r in resultados if r["Algoritmo"] == modelo_principal
        )
        st.markdown(f"#### Resultado destacado: {modelo_principal}")
        st.metric("Probabilidad estimada de clase 1", f"{p_destacado:.1%}")
        if p_destacado >= 0.5:
            st.error("El modelo seleccionado estima la clase 1. Se recomienda consultar a un profesional de salud.")
        else:
            st.success("El modelo seleccionado estima la clase 0.")
        st.caption(
            "La probabilidad es una salida del clasificador y no equivale a riesgo clínico calibrado "
            "ni a un diagnóstico."
        )

# ------------------- COMPARATIVA -------------------
with tab_comp:
    st.subheader("Comparación de los cuatro algoritmos")
    st.caption("Se muestran las métricas registradas en el experimento original de Google Colab.")
    ruta_csv = BASE / "comparativa_examen.csv"
    if ruta_csv.exists():
        comparativa = pd.read_csv(ruta_csv)
        conjunto = st.radio("Conjunto de evaluación", ["Prueba", "Entrenamiento"], horizontal=True)
        datos = comparativa[comparativa["Conjunto"] == conjunto].copy()
        columnas_metricas = ["Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"]
        st.dataframe(datos[["Algoritmo"] + columnas_metricas].style.format({c: "{:.3f}" for c in columnas_metricas}), use_container_width=True, hide_index=True)
        st.markdown("#### Comparación visual")
        st.bar_chart(datos.set_index("Algoritmo")[columnas_metricas])
    else:
        st.warning("No se encontró comparativa_examen.csv.")

    st.markdown("#### Gráficas exportadas desde el notebook")
    for archivo, titulo in [
        ("roc_curvas.png", "Curvas ROC de los cuatro algoritmos"),
        ("matrices_confusion.png", "Matrices de confusión"),
        ("curvas_aprendizaje.png", "Curvas de rendimiento y pérdida del MLP"),
    ]:
        ruta = BASE / archivo
        if ruta.exists():
            with st.expander(titulo, expanded=(archivo == "roc_curvas.png")):
                st.image(str(ruta), use_container_width=True)

# ------------------- DATOS -------------------
with tab_datos:
    st.subheader("Conjunto de datos Cleveland")
    a, b, c = st.columns(3)
    a.metric("Pacientes", len(df))
    b.metric("Variables predictoras", len(FEATURES))
    c.metric("Columnas", df.shape[1])
    conteo = df["diagnosis"].value_counts().reindex([0, 1], fill_value=0)
    x, ycol = st.columns(2)
    x.metric("Clase 0 - sin enfermedad", int(conteo.loc[0]))
    ycol.metric("Clase 1 - enfermedad", int(conteo.loc[1]))
    st.markdown("#### Primeros registros")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("#### Descripción estadística")
    st.dataframe(df.describe().T, use_container_width=True)

st.divider()
st.caption(
    "Proyecto académico del Grupo 4. Dataset Cleveland Heart Disease. "
    "Las salidas de los modelos son experimentales y no deben utilizarse para decisiones médicas."
)
