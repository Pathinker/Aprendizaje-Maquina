import io
import sys
import random
import pandas as pd
import seaborn as sns # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Permitir la representación de acentos,

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

x1Etiqueta = "ENGINE SIZE"
x2Etiqueta = "FUEL CONSUMPTION"
x3Etiqueta = "COEMISSIONS "

x1 = dataFrame[x1Etiqueta] # x
x2 = dataFrame[x2Etiqueta] # y
x3 = dataFrame[x3Etiqueta] # z

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

plt.title("Regresión Múltiple Lineal")
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

# Desempaquetar nuevamente los datos de entrenamiento.

x1 = dataFrameEntrenamiento[x1Etiqueta] # x
x2 = dataFrameEntrenamiento[x2Etiqueta] # y
x3 = dataFrameEntrenamiento[x3Etiqueta] # z

valoresIndependientes =  np.column_stack((x1, x2))

modeloValidacionCruzada = LinearRegression()
modeloValidacionCruzada = modeloValidacionCruzada.fit(valoresIndependientes, x3)

# Desempaquetar los datos de evaluación.

x1Prediccion = dataFrameEntrenamiento[x1Etiqueta] # x
x2Prediccion = dataFrameEntrenamiento[x2Etiqueta] # y

valoresPrediccion = np.column_stack((x1Prediccion, x2Prediccion))

x1Evaluacion = dataFrameEvaluacion[x1Etiqueta] # x
x2Evaluacion= dataFrameEvaluacion[x2Etiqueta] # y
x3Evaluacion = dataFrameEvaluacion[x3Etiqueta]

valoresEvaluacion = np.column_stack((x1Evaluacion, x2Evaluacion))

print("\n--- Entrenamiento 80% Evaluacion 20% Validacion Cruzada ---\n\n" 
      "Intercepto: {}\n"
      "Coeficientes: {}\n"
      "Error Cuadratico Medio: {}\n"
      "Precisión: {}".format(modeloValidacionCruzada.intercept_, 
                             modeloValidacionCruzada.coef_, 
                             mean_squared_error(x3, modeloValidacionCruzada.predict(valoresIndependientes)),
                             r2_score(dataFrameEvaluacion[x3Etiqueta], modeloValidacionCruzada.predict(valoresEvaluacion))))

# Graficar el plano resusltante.

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

# Graficar los datos del dataset

ax.scatter(valoresPrediccion[:, 0], valoresPrediccion[:, 1], x3, c = "blue", marker = "o", label = "Valores Entrenamiento")
xx, yy = np.meshgrid(np.linspace(np.min(valoresPrediccion[:, 0]), np.max(valoresPrediccion[:, 0]), 100),
                     np.linspace(np.min(valoresPrediccion[:, 1]), np.max(valoresPrediccion[:, 1]), 100))

ax.scatter(valoresEvaluacion[:, 0], valoresEvaluacion[:, 1], x3Evaluacion, c = "green", marker = "o", label = "Valores Evaluación")

# Ecuación de la regresión lineal múltiple donde el plano obtenido, es una extensión de la ecuación de una recta convencional
# Donde son incorporados todas las pendientes que multiplicaran al dato proporcionado más un sesgo.

zz = modeloValidacionCruzada.intercept_ + modeloValidacionCruzada.coef_[0] * xx + modeloValidacionCruzada.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color = "r", alpha = 0.2)

ax.set_xlabel("Tamaño Motor")
ax.set_ylabel("Consumo Combustible")
ax.set_zlabel("Emisiones Carbono")
ax.azim = 60
ax.elev = 10

plt.title("Regresión Múltiple Lineal: Validación Cruzada")
plt.legend()
plt.show()