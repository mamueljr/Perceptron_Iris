import joblib
import pandas as pd
from sklearn.datasets import load_iris
import streamlit as st

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
    print(features)


# modelo_entrenado = joblib.load('model_perceptron.sav')
# pipeline=joblib.load('pipeline_perceptron.sav')
