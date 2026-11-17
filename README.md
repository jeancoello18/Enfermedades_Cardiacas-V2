# Dashboard cardiovascular - versión optimizada para Streamlit Community Cloud

## Cambios para evitar el inicio infinito
- TensorFlow ya no se importa al iniciar la aplicación.
- Los modelos se entrenan/cargan solo cuando el usuario pulsa **Evaluar con los cuatro algoritmos**.
- Los modelos quedan guardados en la caché de Streamlit durante la vida de la instancia.
- La pestaña Comparativa utiliza las métricas y gráficas exportadas desde Colab, sin volver a entrenar modelos.

## Archivos
Todos deben estar en la misma carpeta que `app.py`: `processed.cleveland.data`, `modelo_final.joblib`, `scaler.joblib`, `info_modelo.json`, `comparativa_examen.csv`, `roc_curvas.png`, `curvas_aprendizaje.png`, `matrices_confusion.png` y `requirements.txt`.

## Despliegue
1. Reemplaza en GitHub los archivos de la aplicación por los de este ZIP.
2. En Streamlit Community Cloud configura `app.py` como Main file path.
3. Espera a que termine la instalación de dependencias.
4. Abre la app. La interfaz debe mostrarse sin esperar el entrenamiento del MLP.
5. Al realizar la primera predicción, se entrenan RF1, Regresión logística y MLP, y se carga RF2. Las predicciones posteriores reutilizan los modelos en caché.

## Nota
La primera predicción puede tardar por el entrenamiento del MLP. Si el entorno gratuito se queda sin memoria, exporta el MLP entrenado desde Colab (por ejemplo, formato `.keras`) y ajusta la aplicación para cargarlo, en vez de entrenarlo en la nube.

Uso académico. No es un dispositivo médico ni proporciona diagnósticos.
