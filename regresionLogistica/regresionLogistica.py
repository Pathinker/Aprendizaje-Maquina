import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix

class regresionLogistica:

    def __init__(self, learningRate=0.01, epochs=1000):
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z): # Función lógistica o sigmoideal para clasificación binaria
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y): # Función de perdida "Binary Cross Entropy"
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):

        # Inicializar los parámetros del modelo

        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradiente descendente para actualizar los parámetros
        for _ in range(self.epochs):

            # Forward Propagation
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Backward propagation
            loss = self.loss(y_predicted, y)
            self.losses.append(loss)

            # Calcular gradientes
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Actualizar los parámetros
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

# Reconocimiento de vinos, emplea 13 atributos para reconocer 3 clases diferentes de vinos, no obstante sera reducido a 2.

datasetVino = datasets.load_wine()
clasesVino = datasetVino.data[:, :2] # Seleccionar unicamente dos clases
labelsVino =  (datasetVino.target  != 0) * 1 # Transformar los labels o etiquetas a valores binarios.

# Aplicar Cross Validation para ello definimos la cantidad de folders y el numero de folders de validacion

numeroFolders = 10
numeroFoldersValidacion = 2
random.seed(42) # Valor arbitrario para reproducir el ordenamiento aleatorio en cada ejecución del programa.

# Para extraer los datos en cada una de las carpetas guardaremos todos los indices en un solo array para posteriormente dividirlo en la cantidad de carpetas especificadas.
# Consecuentemente son seleccionada las carpetas a su clasificación correspondiente.

indices = np.array(range(len(clasesVino))) # Generar una lista de números consecutivos del tamaño de elementos.
random.shuffle(indices) # Aleatorizar los indices generados.
indicesFolders = np.array_split(indices, numeroFolders) # Es preferible usar arra_split ya que garantiza la division de elementos cuando las cantidades no son exactas

# Seleccionar aleatoriamente los folders a cargar y los concatena en un np.array

foldersTest = np.random.choice(len(indicesFolders), numeroFoldersValidacion, replace = False)

indicesTest = np.concatenate([indicesFolders[i] for i in foldersTest])

# Lo mismo es adecuado con la lista de entranmiento, no obstante es necesario verificar que no esten presentes la anterior categoria.

listaEntrenamiento = []

for i in range(len(indicesFolders)):

    if i not in foldersTest:
        
        listaEntrenamiento.extend(indicesFolders[i])

indicesEntrenamiento = np.array(listaEntrenamiento)

# Cargar de manera correspondiente los valores de entrenamiento y validación, en conjunto con sus labes correspondientes.

xTrain = [clasesVino[i] for i in indicesEntrenamiento]
xTest = [clasesVino[i] for i in indicesTest]
yTrain = [labelsVino[i] for i in indicesEntrenamiento]
yTest = [labelsVino[i] for i in indicesTest]

# Transformar a np.array para evitar incompatibilidades con sklearn.

xTrain = np.array(xTrain)
xTest = np.array(xTest)
yTrain = np.array(yTrain)
yTest = np.array(yTest)

# Crear y entrenar el modelo

model = regresionLogistica(learningRate=0.01, epochs=1000)
model.fit(xTrain, yTrain)

# Hacer predicciones en el conjunto de prueba
Predicciones = model.predict(xTest)

# Calcular la matriz de confusión
cm = confusion_matrix(yTest, Predicciones)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xticks([0, 1], ['Vino Tipo A', 'Vino Tipo B'])
plt.yticks([0, 1], ['Vino Tipo A', 'Vino Tipo B'])
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.show()

print(np.array(cm))