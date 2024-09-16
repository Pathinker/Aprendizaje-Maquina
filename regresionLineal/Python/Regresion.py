import io
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Permitir la representación de acentos, porque uso otro metodo de encoding en el archivo CSV

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def regresionLineal(Datos):

    sumatoriaX = 0
    sumatoriaY = 0
    sumatoriaProductos = 0
    sumatoriaXCuadrada = 0

    for i in range(len(Datos)):

        sumatoriaX += Datos.iloc[i, 0]
        sumatoriaY += Datos.iloc[i, 1]
        sumatoriaProductos += Datos.iloc[i, 0] * Datos.iloc[i, 1]
        sumatoriaXCuadrada += Datos.iloc[i, 0] ** 2

    m = ((len(Datos) * sumatoriaProductos) - (sumatoriaX * sumatoriaY)) / ((len(Datos) * sumatoriaXCuadrada) - (sumatoriaX ** 2))
    b = (sumatoriaY -  m * sumatoriaX) / len(Datos)

    promedioX = sumatoriaX / len(Datos)
    promedioY = sumatoriaY / len(Datos)
    diferenciaXY = 0
    diferenciaXCuadrada = 0
    diferenciaYCuadrada = 0

    for i in range(len(Datos)):

        diferenciaXY += (Datos.iloc[i, 0] - promedioX) * (Datos.iloc[i, 1] - promedioY)

        diferenciaXCuadrada += (Datos.iloc[i, 0] - promedioX) ** 2
        diferenciaYCuadrada += (Datos.iloc[i, 1] - promedioY) ** 2

    coeficienteCorrelacion = diferenciaXY / ((diferenciaXCuadrada ** (1/2)) * (diferenciaYCuadrada ** (1/2)))

    return m, b, coeficienteCorrelacion

def graficarInformacion(datosEntrenamiento, datosPrediccion = None):

    m, b, coeficienteCorrelacion = regresionLineal(datosEntrenamiento)

    x = []
    y = []
    prediccion = []
    erroresPrediccion = 0

    for i in range(len(datosEntrenamiento)):

        prediccion.append(datosEntrenamiento.iloc[i, 0] * m + b)

        x.append(datosEntrenamiento.iloc[i, 0])
        y.append(datosEntrenamiento.iloc[i, 1])

        erroresPrediccion += (x[i] - prediccion[i]) ** 2

    erroresPrediccion /= len(datosEntrenamiento)

    print("\nPendiente: {} \nCoordenada Origen (Y): {} \nCoeficiente de Correlación: {} \nError Cuadrático Medio (MSE): {}".format(m, b, coeficienteCorrelacion, erroresPrediccion))

    plt.title("Entrenamiento del Modelo")
    plt.scatter(x, y, color = "Blue", label = "Datos")
    plt.xlabel("Salario Anual")
    plt.plot(x, prediccion, color = "Red", label = "Regresión Lineal")
    plt.ylabel("Costo Vehiculo")
    plt.legend()

    plt.show()

    if datosPrediccion is not None:
                    
        x = []
        y = []
        prediccion = []
        erroresPrediccion = 0

        for i in range(len(datosPrediccion)):

            prediccion.append(datosPrediccion.iloc[i, 0] * m + b)

            x.append(datosPrediccion.iloc[i, 0])
            y.append(datosPrediccion.iloc[i, 1])
            erroresPrediccion += (x[i] - prediccion[i]) ** 2

        erroresPrediccion /= len(datosPrediccion)

        print("Error Cuadrático Medio (MSE) con Datos no Conocidos: ", erroresPrediccion)
    
        plt.title("Desempeño del Modelo con Datos No Conocidos")
        plt.scatter(x, y, color = "Blue", label = "Datos")
        plt.xlabel("Salario Anual")
        plt.scatter(x, prediccion, color = "Red", label = "Regresión Lineal")
        plt.ylabel("Costo Vehiculo")
        plt.legend()

        plt.show()

# Leer el CSV, solamente adquirir las columnas de información 5 y 8 que son las que contienen la información que necesito.

datos = pd.read_csv(r"regresionLineal/Python/car_purchasing.csv", 
                    delimiter=",", 
                    usecols=[5, 8], 
                    encoding='ISO-8859-1')

print("Caso A: 100% Datos Entrenamiento y Predicción")
graficarInformacion(datos)

# Mezclara aleatoriamente los datos

datos = datos.sample(frac=1).reset_index(drop=True) 

# Separar en 60% Entrenamiento 40% Predicción

numeroEntranamiento =  int(len(datos) * 0.6)
arrayEntrenamiento = datos[:numeroEntranamiento]
arrayPrediccion = datos[numeroEntranamiento:]

print("\nCaso B: 60% Datos Entrenamiento y 40% Predicción")
graficarInformacion(arrayEntrenamiento, arrayPrediccion)