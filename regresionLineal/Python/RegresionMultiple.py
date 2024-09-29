import random
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataFrame = pd.read_csv(r"regresionLineal/Python/FuelConsumption.csv", 
                    delimiter=",")

# Selecionar unicamente las columnas que son numericas y graficar sus coeficientes de correlación, a fin de seleccionar 2 variables correlacionadas con una sola.

dataFrame = dataFrame.select_dtypes(include=["number"])
dataFrame.drop(columns=["Year"], inplace = True) # Remover la columna año del dataset

# Imprimir las columnas disponibles.

Columnas = dataFrame.columns.tolist()
print("Columnas del dataset dispinibles: \n", Columnas)

plt.figure(figsize = (10, 8))
sns.heatmap(dataFrame.corr(), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
plt.show()

# Extraer las variables correlacionadas con la desceada a predecir.

x1 = dataFrame["ENGINE SIZE"] # x
x2 = dataFrame["FUEL CONSUMPTION"] # y
x3 = dataFrame["COEMISSIONS "] # z

valoresIndependientes =  np.column_stack((x1, x2))

modeloSimple = LinearRegression()
modeloSimple.fit(valoresIndependientes, x3)

# Mostrar resultados de entrenamiento

print("\n--- Entrenamiento 100% ---\n\n" 
      "Intercepto: {}\n"
      "Coeficientes: {}\n"
      "Error Cuadratico Medio: {}".format(modeloSimple.intercept_, modeloSimple.coef_, mean_squared_error(x3, modeloSimple.predict(valoresIndependientes))))

# Graficar el plano resusltante.

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

# Graficar los datos del dataset

ax.scatter(valoresIndependientes[:, 0], valoresIndependientes[:, 1], x3, c = "blue", marker = "o")
xx, yy = np.meshgrid(np.linspace(np.min(valoresIndependientes[:, 0]), np.max(valoresIndependientes[:, 0]), 100),
                     np.linspace(np.min(valoresIndependientes[:, 1]), np.max(valoresIndependientes[:, 1]), 100))

# Ecuación de la regresión lineal múltiple donde el plano obtenido, es una extensión de la ecuación de una recta convencional
# Donde son incorporados todas las pendientes que multiplicaran al dato proporcionado más un sesgo.

zz = modeloSimple.intercept_ + modeloSimple.coef_[0] * xx + modeloSimple.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color = "r", alpha = 0.2)

ax.set_xlabel("Tamaño Motor")
ax.set_ylabel("Consumo Combustible")
ax.set_zlabel("Emisiones Carbono")
ax.azim = 60
ax.elev = 10

plt.show()

# Evaluación del modelo haciendo uso de Validación Cruzada 80% Entrenamiento 20% Evaluacion

validacionCruzada = np.array_split(dataFrame, 10)
random.shuffle(validacionCruzada)

dataFrameEntrenamiento = []
dataFrameEvaluacion = []

for i in range(2):

    dataFrameEvaluacion.append(validacionCruzada.pop(random.randint(0, len(validacionCruzada) - 1)))

dataFrameEntrenamiento = pd.concat(validacionCruzada, ignore_index=True)
dataFrameEvaluacion = pd.concat(dataFrameEvaluacion, ignore_index=True)

x1 = dataFrameEntrenamiento["ENGINE SIZE"] # x
x2 = dataFrameEntrenamiento["FUEL CONSUMPTION"] # y
x3 = dataFrameEntrenamiento["COEMISSIONS "] # z

valoresIndependientes =  np.column_stack((x1, x2))

modeloValidacionCruzada = LinearRegression()
modeloValidacionCruzada = modeloValidacionCruzada.fit(valoresIndependientes, x3)

x1Prediccion = dataFrameEntrenamiento["ENGINE SIZE"] # x
x2Prediccion = dataFrameEntrenamiento["FUEL CONSUMPTION"] # y

valoresPrediccion = np.column_stack((x1Prediccion, x2Prediccion))

print("\n--- Entrenamiento 80% Evaluacion 20% Validación Cruzada ---\n\n" 
      "Intercepto: {}\n"
      "Coeficientes: {}\n"
      "Error Cuadratico Medio: {}".format(modeloValidacionCruzada.intercept_, modeloValidacionCruzada.coef_, mean_squared_error(x3, modeloValidacionCruzada.predict(valoresIndependientes))))

# Graficar el plano resusltante.

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

# Graficar los datos del dataset

ax.scatter(valoresPrediccion[:, 0], valoresPrediccion[:, 1], x3, c = "blue", marker = "o")
xx, yy = np.meshgrid(np.linspace(np.min(valoresPrediccion[:, 0]), np.max(valoresPrediccion[:, 0]), 100),
                     np.linspace(np.min(valoresPrediccion[:, 1]), np.max(valoresPrediccion[:, 1]), 100))

# Ecuación de la regresión lineal múltiple donde el plano obtenido, es una extensión de la ecuación de una recta convencional
# Donde son incorporados todas las pendientes que multiplicaran al dato proporcionado más un sesgo.

zz = modeloValidacionCruzada.intercept_ + modeloValidacionCruzada.coef_[0] * xx + modeloValidacionCruzada.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color = "r", alpha = 0.2)

ax.set_xlabel("Tamaño Motor")
ax.set_ylabel("Consumo Combustible")
ax.set_zlabel("Emisiones Carbono")
ax.azim = 60
ax.elev = 10

plt.show()