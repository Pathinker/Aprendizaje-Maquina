import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

imagen = cv2.imread('SOM/Perro.jpg')
imagenRGB = cv2.cvtColor(cv2.imread('SOM/Perro.jpg'), cv2.COLOR_BGR2RGB)
#data = cv2.resize(imagenRGB, (227, 227)) / 255.0

data = imagenRGB / 255.0

somX, somY = 100, 100
inputDimension = 3
weights = np.random.rand(somX, somY, inputDimension)

def findBMU(inputVector, weights):

    distances = np.linalg.norm(weights - inputVector, axis=2)
    bmuIndex = np.unravel_index(np.argmin(distances, axis = None), distances.shape)

    return bmuIndex

def updateWeights(inputVector, weights, bmuIndex, t, maxIter, initLearningRate = 0.5, initRadius = None):

    if(initRadius is None):

        initRadius = max(somX, somY) / 2

    learningRate = initLearningRate * (1 - t / maxIter)
    radius = initRadius * (1 - t / maxIter)

    for i in range(somX):

        for j in range(somY):

            distanceToBMU = np.sqrt((i - bmuIndex[0]) ** 2 + (j - bmuIndex[1]) ** 2)

            if (distanceToBMU <= radius):

                influence = np.exp(- distanceToBMU ** 2 / (2 * (radius ** 2)))

                weights[i, j] += learningRate * influence * (inputVector - weights[i, j])

iteraciones = 10000

for i in tqdm(range(iteraciones), desc="Entrenamiento SOM", unit = "Iteracion", dynamic_ncols=True):

    inputVector = data[np.random.randint(0, data.shape[0]), np.random.randint(0, data.shape[1]), :]
    
    bmuIndex = findBMU(inputVector, weights)
    updateWeights(inputVector, weights, bmuIndex, i, iteraciones, initLearningRate = 0.1)

fig, ax = plt.subplots(figsize=(8, 8))

for i in range(somX):

    for j in range(somY):

        weight = weights[i, j]
        ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=weight))

ax.set_xlim([0, somX])
ax.set_ylim([0, somY])
ax.set_title("Mapa SOM de Colores RGB")
plt.axis('off')
plt.show()