import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import KFold # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score  # type: ignore  # type: ignore
from collections import Counter
from tqdm import tqdm  # Importar tqdm para la barra de carga

class KNeighborsClassifier:

    def __init__(self, vecinos = 5):

        self.vecinos = vecinos

    def fit(self, X, y):

        self.xTrain = X
        self.yTrain = y

    def predict(self, X):

        prediciones = [self._predict(x) for x in X]
        return np.array(prediciones)

    def _predict(self, x):

        # Calcula distancias desde el punto x hasta todos los puntos en el conjunto de entrenamiento.
        distances = [self.distanciaEuclidiana(x, xTrain) for xTrain in self.xTrain]

        # Obtiene los índices de los K vecinos más cercanos.
        kIndices = np.argsort(distances)[:self.vecinos]

        # Obtiene las etiquetas de los K vecinos más cercanos.
        kNearest = [self.yTrain.iloc[i] for i in kIndices]

        # Realiza una votación para determinar la clase más común.
        most_common = Counter(kNearest).most_common(1)

        return most_common[0][0]

    def distanciaEuclidiana(self, x1, x2):

        return np.sqrt(np.sum((x1 - x2) ** 2))
    
# Cargar dataset Penguins y contemplar información básica.

dataset = sns.load_dataset("penguins")
dataset.describe()

# Mostrar en una grafico de barras el número de elementos por clasificación.

especies = dataset["species"].value_counts()
colores = plt.cm.Blues(especies / especies.max())
especies.plot(kind="bar", color=colores)

plt.title("Numero Ejemplares Especie Pingüinos")
plt.xticks(rotation = 0, ha = "center")
plt.xlabel("")
plt.show()

# Mostrar la matriz de correlación cocn los valores númericos disponibles.

datosNumericos = dataset.select_dtypes(include=[np.number])
sns.heatmap(datosNumericos.corr(), square = True, annot = True, cmap='Blues')
plt.title("Matriz de Correlación")
plt.xticks(rotation = 75)
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2) 
plt.show()

# Remover los labels no númericos y de clasificación.

Y = dataset["species"]
X = dataset.drop(columns=["species", "island", "sex"])

datasetKF = KFold(n_splits = 10, shuffle = True, random_state = 9)

# Rango de valores de K.
valoresK = range(1, 21)
presicionMedia = []

# Evaluar cada valor de K.
for k in tqdm(valoresK, desc="Evaluando K"):

    precisiones = []
    
    # Entrenamiento y validación con K-Fold.
    for train_index, test_index in datasetKF.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # Crear y entrenar el clasificador KNN.
        knn = KNeighborsClassifier(vecinos=k)
        knn.fit(X_train.values, y_train)
        
        # Hacer predicciones.
        predicciones = knn.predict(X_test.values)
        
        # Calcular la precisión.
        precision = np.mean(predicciones == y_test.values)
        precisiones.append(precision)
    
    # Calcular la precisión media para este valor de K.
    presicionMedia.append(np.mean(precisiones))

# Encontrar el mejor K.
mejorK = valoresK[np.argmax(presicionMedia)]
mejorPrecision = max(presicionMedia)

# Mostrar resultados
print("Mejor k: {}, Precisión media: {}".format(mejorK, mejorPrecision))

# Visualizar las precisiones para cada K.
plt.figure(figsize=(10, 5))
plt.plot(valoresK, presicionMedia, marker='o', linestyle='-', color='b')
plt.title('Precisión Media K fold por valor de k en KNN')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Precisión media')
plt.xticks(valoresK)
plt.grid()
plt.show()

# Entrenar el modelo final con el mejor K.
KNNFinal = KNeighborsClassifier(vecinos=mejorK)
KNNFinal.fit(X.values, Y)  # Usar todo el conjunto de datos para el entrenamiento

# Hacer predicciones con el modelo entrenado.
prediccionesFinal = KNNFinal.predict(X.values)

# Calcular la matriz de confusión.
cm = confusion_matrix(Y, prediccionesFinal, labels=Y.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Y.unique())

# Visualizar la matriz de confusión.
disp.plot(cmap=plt.cm.Blues)
plt.xlabel("")
plt.ylabel("")
plt.title('Matriz de Confusión')
plt.show()

# Calcular métricas
accuracy = accuracy_score(Y, prediccionesFinal)
precision = precision_score(Y, prediccionesFinal, average='weighted')
recall = recall_score(Y, prediccionesFinal, average='weighted')

# Mostrar métricas
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))