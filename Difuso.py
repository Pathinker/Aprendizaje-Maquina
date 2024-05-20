import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir las variables de entrada y salida difusas
volumenVehiculos = ctrl.Antecedent(np.arange(0, 301, 1), 'volumenVehiculos')
horaDia = ctrl.Antecedent(np.arange(0, 25, 1), 'horaDia')
condicionesClimaticas = ctrl.Antecedent(np.arange(0, 51, 1), 'condicionesClimaticas')
horaDia = ctrl.Consequent(np.arange(0, 101, 1), 'potencia_climatizacion')

# Definir las funciones de membresía para las variables de entrada y salida
volumenVehiculos['Bajo'] = fuzz.trapmf(volumenVehiculos.universe, [0, 100])
volumenVehiculos['Medio'] = fuzz.trimf(volumenVehiculos.universe, [50, 125, 200])
volumenVehiculos['Alto'] = fuzz.trapmf(volumenVehiculos.universe, [150, 300])

horaDia['Bajo'] = fuzz.trapmf(volumenVehiculos.universe, [6, 8])
horaDia['Medio'] = fuzz.trimf(volumenVehiculos.universe, [7.5, 8.25, 9])
horaDia['Alto'] = fuzz.trapmf(volumenVehiculos.universe, [18, 20])

# Definir las reglas difusas
regla1 = ctrl.Rule(temperatura_ambiente['frio'], potencia_climatizacion['baja'])
regla2 = ctrl.Rule(temperatura_ambiente['templado'], potencia_climatizacion['media'])
regla3 = ctrl.Rule(temperatura_ambiente['caliente'], potencia_climatizacion['alta'])

# Crear el sistema de control difuso
sistema_control = ctrl.ControlSystem([regla1, regla2, regla3])
sistema_control_difuso = ctrl.ControlSystemSimulation(sistema_control)

# Asignar valores de entrada al sistema de control difuso
sistema_control_difuso.input['temperatura_ambiente'] = 10

# Activar el sistema de control difuso
sistema_control_difuso.compute()

# Obtener el valor de salida del sistema de control difuso
valor_potencia = sistema_control_difuso.output['potencia_climatizacion']

# Mostrar el resultado
print("Valor de potencia de climatización:", valor_potencia)

# Graficar las funciones de membresía y la salida
temperatura_ambiente.view(sim=sistema_control_difuso)
potencia_climatizacion.view(sim=sistema_control_difuso)
