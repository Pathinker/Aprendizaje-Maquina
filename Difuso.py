!pip install scikit-fuzzy > /dev/null 2>&1

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definir las variables de entrada y salida difusas
volumenVehiculos = ctrl.Antecedent(np.arange(0, 301, 1), 'volumenVehiculos')
horaDia = ctrl.Antecedent(np.arange(0, 25, 1), 'horaDia')
condicionesClimaticas = ctrl.Antecedent(np.arange(0, 51, 1), 'condicionesClimaticas')
tiempoSemaforo = ctrl.Consequent(np.arange(10, 61, 1), 'tiempoSemaforo')

# Definir las funciones de membresía para las variables de entrada y salida
volumenVehiculos['Bajo'] = fuzz.trapmf(volumenVehiculos.universe, [0, 0, 50, 100])
volumenVehiculos['Medio'] = fuzz.trimf(volumenVehiculos.universe, [50, 125, 200])
volumenVehiculos['Alto'] = fuzz.trapmf(volumenVehiculos.universe, [150, 200, 300, 300])

horaDia['Temprano'] = fuzz.trapmf(horaDia.universe, [6, 6.25, 7.75, 8])
horaDia['Pico'] = fuzz.trimf(horaDia.universe, [7.5, 8.25, 9])
horaDia['Pico2'] = fuzz.trimf(horaDia.universe, [17, 18, 19])
horaDia['Noche'] = fuzz.trapmf(horaDia.universe, [18, 18.25, 19.75, 20])

condicionesClimaticas['Normal'] = fuzz.trapmf(condicionesClimaticas.universe, [0, 0, 2.5, 5])
condicionesClimaticas['Ligera'] = fuzz.trimf(condicionesClimaticas.universe, [0, 10, 20])
condicionesClimaticas['Fuerte'] = fuzz.trapmf(condicionesClimaticas.universe, [10, 12.5, 50, 50])

tiempoSemaforo['Corto'] = fuzz.trapmf(tiempoSemaforo.universe, [10, 10, 20, 25])
tiempoSemaforo['Medio'] = fuzz.trimf(tiempoSemaforo.universe, [20, 32.5, 45])
tiempoSemaforo['Largo'] = fuzz.trapmf(tiempoSemaforo.universe, [40, 45, 60, 60])

# Definir las reglas difusas
regla1 = ctrl.Rule(volumenVehiculos['Alto'] & (horaDia["Pico"] | horaDia["Pico2"]) & condicionesClimaticas["Fuerte"], tiempoSemaforo['Largo'])
regla2 = ctrl.Rule(volumenVehiculos['Alto'] & horaDia["Temprano"] & condicionesClimaticas["Normal"], tiempoSemaforo['Largo'])
regla3 = ctrl.Rule(volumenVehiculos['Alto'] & horaDia["Noche"] & condicionesClimaticas["Ligera"], tiempoSemaforo['Largo'])
regla4 = ctrl.Rule(volumenVehiculos['Medio'] & (horaDia["Pico"] | horaDia["Pico2"]) & condicionesClimaticas["Fuerte"], tiempoSemaforo['Medio'])
regla5 = ctrl.Rule(volumenVehiculos['Medio'] & (horaDia["Pico"] | horaDia["Pico2"]) & condicionesClimaticas["Normal"], tiempoSemaforo['Medio'])
regla6 = ctrl.Rule(volumenVehiculos['Medio'] & horaDia["Temprano"] & condicionesClimaticas["Ligera"], tiempoSemaforo['Corto'])
regla7 = ctrl.Rule(volumenVehiculos['Bajo'] & (horaDia["Pico"] | horaDia["Pico2"]) & condicionesClimaticas["Fuerte"], tiempoSemaforo['Corto'])
regla8 = ctrl.Rule(volumenVehiculos['Bajo'] & horaDia["Temprano"] & condicionesClimaticas["Normal"], tiempoSemaforo['Corto'])
regla9 = ctrl.Rule(volumenVehiculos['Bajo'] & horaDia["Noche"] & condicionesClimaticas["Ligera"], tiempoSemaforo['Corto'])

# Crear el sistema de control difuso
sistemaControl = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6, regla7, regla8, regla9])
sistemaControlDifuso = ctrl.ControlSystemSimulation(sistemaControl)

# Asignar valores de entrada al sistema de control difuso
sistemaControlDifuso.input['volumenVehiculos'] = 175
sistemaControlDifuso.input['horaDia'] = 18.1 # Tiene que admitir un rango de valores valido, no es posible colocar un valor en la funcion discontinua.
sistemaControlDifuso.input['condicionesClimaticas'] = 5

# Activar el sistema de control difuso
sistemaControlDifuso.compute()

# Obtener el valor de salida del sistema de control difuso
valorSemaforo = sistemaControlDifuso.output['tiempoSemaforo']

# Graficar las funciones de membresía y la salida
print("Factor Pertencia Volumen Vehiculos:  \n")
volumenVehiculos.view(sim=sistemaControlDifuso)
plt.show()

print("\nFactor Pertencia Hora Dia: \n")
horaDia.view(sim=sistemaControlDifuso)
plt.show()

print("\nFactor Pertencia Condiciones Climaticas:  \n")
condicionesClimaticas.view(sim=sistemaControlDifuso)
plt.show()

print("\nValor de potencia del semaforo:", valorSemaforo, "\n")
tiempoSemaforo.view(sim=sistemaControlDifuso)
plt.show()
