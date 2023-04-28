import joblib
import pandas as pd
from sklearn.datasets import load_iris
import streamlit as st
import time

iris = load_iris(as_frame=True)
# Desplegar app
st.title('Predictor para tipo de flor Iris usando perceptron')

st.header('Datos de Entrada:')
sepal_length = st.slider('Sepal length (cm):', 1.0, 8.0, 4.0)
sepal_width = st.slider('Sepal width (cm):', 0.0, 5.0, 3.0)
petal_length = st.slider('Petal length (cm):', 1.0, 8.0, 1.0)
petal_width = st.slider('Petal width (cm):', 0.0, 5.0, 0.4)

if st.button('Iniciar'):
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    # print(features)
    modelo_entrenado = joblib.load('model_perceptron.sav')
    pipeline = joblib.load('pipeline_perceptron.sav')
    transformados = pipeline.transform(features)
    perc_model = modelo_entrenado.predict(transformados)
    # Barra de progreso
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(
            f'Calculando la clasificacion de flor Iris {i+1}%')
        bar.progress(i+1)
        time.sleep(0.001)
    st.subheader(f'Tipo de Iris: {iris.target_names[perc_model[0]]}')
