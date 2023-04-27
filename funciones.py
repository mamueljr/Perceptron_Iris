import os
import joblib


def prediccion(data, model_name):
    model_perceptron = joblib.load('model_perceptron.sav')
    tr_data = joblib.load('pipeline_perceptron.sav')
    tr_pipeline = tr_data.transform(data)
    return model_perceptron.prediccion(tr_pipeline)
