import io
import sys  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Permitir la representación de acentos,

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class RegresionLinealMultiple:

    def __init__(self, learningRate = 0.01, iteraciones = 1000):

        self.learningRate = learningRate
        self.iteraciones = iteraciones
        self.coef_ = None
        self.intercept_= None
    
    def fit(self, X, y):

        XIntercepto = np.column_stack((np.ones(len(X)), X))
        self.coef_ = np.random.rand(XIntercepto.shape[1])

        for _ in range(self.iteraciones):

            gradiente = -2 * XIntercepto.T @ (y - XIntercepto @ self.coef_)
            self.coef_ -= self.learningRate * gradiente
        
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):

        XIntercepto = np.column_stack((np.ones(len(X)), X))
        return XIntercepto @ np.concatenate(([self.intercept_], self.coef_))
    

dataFrame = pd.read_csv(r"regresionLineal/Python/FuelConsumption.csv", 
                    delimiter=",")

# Selecionar unicamente las columnas que son numericas y graficar sus coeficientes de correlación, a fin de seleccionar 2 variables correlacionadas con una sola.

dataFrame = dataFrame.select_dtypes(include=["number"])
dataFrame.drop(columns=["Year"], inplace = True) # Remover la columna año del dataset

# Extraer las variables correlacionadas con la desceada a predecir.

x1 = dataFrame["ENGINE SIZE"] # x
x2 = dataFrame["FUEL CONSUMPTION"] # y
x3 = dataFrame["COEMISSIONS "] # z

# Evitar desbordamiento por overflow

scaler = StandardScaler()
valoresIndependientes = scaler.fit_transform(np.column_stack((x1, x2)))

# Implementar la validación cruzada del modelo para ajustarlo a una n cantidad de Folders seleccionadas

kfolder = 5

modelo = RegresionLinealMultiple(learningRate=0.0001, iteraciones = 10000)
carpetas = np.array_split(valoresIndependientes, kfolder)
valoresEsperados = np.array_split(x3, kfolder)

mseResultados = []
presicionModelo = []

for i in range(kfolder):

    # Mostrar en que kfolder se encuentra el modelo de entrenamiento.

    print("Entrenando Folder Numero: {}".format(i + 1))

    # Seleccionar la carpeta de validación

    validacion = carpetas[i]
    resultadoEsperado = valoresEsperados[i]

    # Asignar nuestros elementos de entrenamiento

    entrenamiento = np.vstack(carpetas[:i] + carpetas[i + 1:])
    entrenamientoResultados = np.concatenate(valoresEsperados[:i] + valoresEsperados[i + 1:])

    # Entrenar

    modelo.fit(entrenamiento, entrenamientoResultados)

    # Validar

    predicciones = modelo.predict(validacion)

    presicion = r2_score(resultadoEsperado, predicciones)

    print("Presición Modelo: {}".format(presicion))
    presicionModelo.append(presicion)

    # Calcular el error cuadrático medio
    mse = mean_squared_error(resultadoEsperado, predicciones)
    mseResultados.append(mse)

    print("Error Cuadrático Medio {}\n".format( mse))

print("\n--- Resultados Finales ---- \n\nCoeficientes: {}\nIntercepto: {}\nDesempeño Modelo: {}\nError Medio {}".format(modelo.coef_, modelo.intercept_,
                                                                                                        sum(presicionModelo) / len(presicionModelo),
                                                                                                        sum(mseResultados) / len(mseResultados)))

# Graficar Rendimiento y Error Modelo por Folder

numeroKFolder = []

for i in range(kfolder):

    numeroKFolder.append(i)

plt.figure()
plt.title("Precisión Modelo")
plt.plot(numeroKFolder, presicionModelo, color = "blue")
plt.axhline(y = max(presicionModelo), linestyle = "--", color = "red")
plt.ylabel("Precisión")
plt.ylim(min(presicionModelo) / 1.3, 1)
plt.xlabel("K Folder")
plt.plot()

plt.figure()
plt.title("Error Modelo")
plt.plot(numeroKFolder, mseResultados, color = "blue")
plt.axhline(y = min(mseResultados), linestyle = "--", color = "red")
plt.ylabel("Error")
plt.xlabel("K Folder")
plt.plot()

# Graficar los datos del dataset

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

ax.scatter(valoresIndependientes[:, 0], valoresIndependientes[:, 1], x3, c = "blue", marker = "o")
xx, yy = np.meshgrid(np.linspace(np.min(valoresIndependientes[:, 0]), np.max(valoresIndependientes[:, 0]), 100),
                     np.linspace(np.min(valoresIndependientes[:, 1]), np.max(valoresIndependientes[:, 1]), 100))

# Ecuación de la regresión lineal múltiple donde el plano obtenido, es una extensión de la ecuación de una recta convencional
# Donde son incorporados todas las pendientes que multiplicaran al dato proporcionado más un sesgo.

zz = modelo.intercept_ + modelo.coef_[0] * xx + modelo.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color = "r", alpha = 0.2)

ax.set_xlabel("Tamaño Motor")
ax.set_ylabel("Consumo Combustible")
ax.set_zlabel("Emisiones Carbono")
ax.azim = 60
ax.elev = 10

plt.title("Regresión Múltiple Lineal: Validación Cruzada K Folders")
plt.show()