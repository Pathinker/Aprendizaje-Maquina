import numpy as np
import matplotlib.pyplot as plt

def regresionLineal(Datos):

    sumatoriaX = 0
    sumatoriaY = 0
    sumatoriaProductos = 0
    sumatoriaXCuadrada = 0

    for i in range(len(Datos)):

        sumatoriaX += Datos[i][0]
        sumatoriaY += Datos[i][1]
        sumatoriaProductos += Datos[i][0] * Datos[i][1]
        sumatoriaXCuadrada += Datos[i][0] ** 2

    m = ((len(Datos) * sumatoriaProductos) - (sumatoriaX * sumatoriaY)) / ((len(Datos) * sumatoriaXCuadrada) - (sumatoriaX ** 2))
    b = (sumatoriaY -  m * sumatoriaX) / len(Datos)

    promedioX = sumatoriaX / len(Datos)
    promedioY = sumatoriaY / len(Datos)
    diferenciaXY = 0
    diferenciaXCuadrada = 0
    diferenciaYCuadrada = 0

    for i in range(len(Datos)):

        diferenciaXY += (Datos[i][0] - promedioX) * (Datos[i][1] - promedioY)

        diferenciaXCuadrada += (Datos[i][0] - promedioX) ** 2
        diferenciaYCuadrada += (Datos[i][1] - promedioY) ** 2

    coeficienteCorrelacion = diferenciaXY / ((diferenciaXCuadrada ** (1/2)) * (diferenciaYCuadrada ** (1/2)))

    return m, b, coeficienteCorrelacion

def graficarInformacion(Datos):

    m, b, coeficienteCorrelacion = regresionLineal(Datos)

    x = []
    y = []

    for i in range(len(Datos)):

        x.append(Datos[i][0])
        y.append(Datos[i][1])

    plt.scatter(x, y, color = "Blue")
    #plt.scatter(x, prediccion, color = "Red")

    plt.show()

# Leer el CSV, solamente adquirir las columnas de información 5 y 8 que son las que contienen la información que necesito.

datos = np.loadtxt(r"regresionLineal/Python/car_purchasing.csv", 
                   skiprows = 1,  
                   delimiter = ",", 
                   usecols = [5, 8],
                   dtype = float)

graficarInformacion(datos)

numeroEntranamiento =  int(len(datos) * 0.6)

np.random.shuffle(datos)
arrayEntrenamiento = datos[:numeroEntranamiento]
arrayPrediccion = datos[numeroEntranamiento:]

print(regresionLineal(arrayEntrenamiento))