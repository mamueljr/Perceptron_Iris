#import joblib
#import os
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import time
from funciones import predict


iris = load_iris(as_frame=True)
# Despliegue de la app
st.title('Predictor para tipo de flor Iris')
# Lateral
st.sidebar.header('Datos de Entrada:')
sepal_length = st.sidebar.slider('Sepal length (cm):', 1.0, 8.0, 4.0)
sepal_width = st.sidebar.slider('Sepal width (cm):', 0.0, 5.0, 3.0)
petal_length = st.sidebar.slider('Petal length (cm):', 1.0, 8.0, 1.0)
petal_width = st.sidebar.slider('Petal width (cm):', 0.0, 5.0, 0.4)


st.header('Perceptron')
model_option = st.header('Modelo a usar:', 'Perceptron')

if st.button('Predecir'):
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    pctn = predict(features,model_option)
    #lr = predict(features, model_option)
    #svc = predict(features, model_option)
    #dt = predict(features, model_option)
    #vc = predict(features, model_option)

    # Barra de progreso
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(
            f'Calculando la clasificacion de flor Iris {i+1}%')
        bar.progress(i+1)
        time.sleep(0.01)
    result = predict(features, model_option)
    st.subheader(f'La clasificacion de la flor Iris es: {iris.target_names[pctn[0]]}')

