from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
# Cargar Iris
iris=datasets.load_iris()

# Features y labels
x = iris.data
y = iris.target
# Split en training y test
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
# standarizar la data
sc = StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
# Escoger el algoritmo para pasar los hiperparametros
ppn = Perceptron(max_iter=1000, eta0= 0.0001, random_state=0, tol=0.001)
# entrenar el modelo
ppn.fit(X_train_std,y_train)
# Hacer la prediccion
y_pred = ppn.predict(X_train_std)
print('-----------')
print((y_test != y_pred).sum(),'/',((y_test == y_pred).sum()+(y_test != y_pred).sum()))
print('Accuracy: ',100*accuracy_score(y_test,y_pred))
