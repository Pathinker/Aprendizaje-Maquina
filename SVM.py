import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn import datasets # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score # type: ignore
from sklearn.decomposition import PCA # type: ignore

# Cargar los datos de entrenamiento y segmentarlos.

datasetVino = datasets.load_wine()
atributos = datasetVino.data
objetivo = datasetVino.target

# Definir nombres de las clases
nombreClases = ['Vino 1', 'Vino 2', 'Vino 3']

xTrain, xTest, yTrain, yTest = train_test_split(atributos, objetivo, test_size = 0.15, random_state = 30)

SVM = SVC(kernel = "linear")

# Encontar el mejor valor de C para la creación del hiperplano y el max margin.

hiperpametros = {"C" : [0.1, 0.5, 0.6, 0.7, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 10, 100, 1000]}
mejorC = GridSearchCV(SVM, hiperpametros, cv = 5, scoring = "accuracy")
mejorC.fit(xTrain, yTrain)

# Mejor modelo encontrado
print(f"Mejor valor de C: {mejorC.best_params_}")
print(f"Mejor precisión: {mejorC.best_score_}")

# Evaluar el mejor modelo en los datos de prueba
mejorSVM = mejorC.best_estimator_
predicciones = mejorSVM.predict(xTest)

# Validación cruzada con k-fold (k=10)
kFold = cross_val_score(mejorSVM, atributos, objetivo, cv=10)
print(f"Resultados de validación cruzada (k=10): {kFold.tolist()}")
print(f"Precisión promedio de validación cruzada: {np.mean(kFold)}")

# Mapa de calor para la matriz de confusión
def matrizConfusion(matriz, nombres):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, yticklabels = nombres, xticklabels = nombreClases)
    plt.title("Matriz de Confusión")
    plt.show()

# Llamada a la función para graficar la matriz de confusión
matrizConfusion(confusion_matrix(yTest, predicciones), nombreClases)

# Reducir dimensionalidades usando PCA para visualizar la división lineal del SVM
pca = PCA(n_components=2)
xTrainPca = pca.fit_transform(xTrain)

# Entrenamiento de SVM en el espacio reducido
mejorSVM.fit(xTrainPca, yTrain)

# Generar un grid para el gráfico y contemplar la frontera de desiciones, dicha fronteras es obtenida al evaluar en cada punto del grid el valor correspondiente retornado por el modelo.
x_min, x_max = xTrainPca[:, 0].min() - 1, xTrainPca[:, 0].max() + 1
y_min, y_max = xTrainPca[:, 1].min() - 1, xTrainPca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

# Predicciones del modelo en el espacio de grid
Z = mejorSVM.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualización de la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)

scatter = plt.scatter(xTrainPca[:, 0], xTrainPca[:, 1], c=yTrain, s=40, edgecolor='k', cmap=plt.cm.coolwarm)

# Añadir leyenda
legend = plt.legend(handles=scatter.legend_elements()[0], labels = nombreClases, title="Clases")
plt.gca().add_artist(legend)

plt.xticks([])
plt.yticks([])
plt.title("Frontera de decisión SVM con PCA (Reducción Dimensiones)")
plt.show()