from email.policy import default
import time
from matplotlib import axis
import pandas as pd
import numpy as np
import random
import logging
from sklearn.metrics import davies_bouldin_score

N_CLUSTERS = 2
NUMERO_IT = 100
CORTE = 0.01
COLUMS = [1, 3, 4]


class KMeans:

    distance = "euclidean"

    def __init__(self, df, colums, k=N_CLUSTERS, max_it=NUMERO_IT, corte=CORTE) -> None:
        self.colums = colums
        self.max_it = max_it
        self.corte = corte
        self.k = k

        # Crear las listas vacias de centroides
        # y la lista de clusters vacios
        self.centroids: list[tuple[float, ...]] = [(0.0,)*k for _ in range(k)]
        self.clusters = [[] for _ in range(k)]

        # Obtener las columnas de interes
        # y obtener k ejemplos aleatorios (diferentes entre si)
        self.df = df.iloc[:, colums]
        examples = random.sample(k=k, population=list(self.df.values))
        logging.info(examples)

        # Añadir a CLUSTERS las dos instancias escogidas aleatoriamente
        # y calcular sus centroides, despues limpiar los clusters para empezar
        # el algoritmo
        for i, ex in enumerate(examples):
            self.clusters[i].append(ex)
        self.centroids = self.__compute_centroids()
        for cluster in self.clusters:
            cluster.clear()

        logging.info(f"Initial centroids: {self.centroids}")
        logging.info(f"Initial cluster: {self.clusters} ")
        print("\n\n")

    def compute(self):
        condition = True
        self.it = 0
        self.start = time.time()
        while condition:
            print("\n\n\n")
            logging.info(f"It {self.it}")

            # Calcular la primera columna de distancias euclidea con el centroide 0
            distancias = self.__compute_distance(self.df.values, self.centroids[0])

            # Ir añadiendo las columnas de los otros centroides
            for centroid in self.centroids[1:]:
                distancias_i = self.__compute_distance(self.df.values, centroid)
                distancias = np.column_stack([distancias, distancias_i])

            # De cada fila, obtener el indice del minimo
            # Guardarlo en la clase para despues calcular el
            # "Bouluar index" mas conocido como "Davies-Bouldin index"
            self.minimos = np.argmin(distancias, axis=1)
            """
            Obtendriamos lo siguiente
            [
                [12.9, 13.8, 4.22],   -> (El menor esta en el indice 2)
                [14.5, 5.5, 5.7],     -> (El menor esta en el indice 1)
                ...
            ]

            [
                2,
                1,
                ...
            ]
            """

            # Como ya sabemos de cual cluster esta mas cerca cada ejemplo
            # con lo obtenido en minimos, añadir cada ejemplo a su cluster
            for i, example in enumerate(self.df.values):
                self.clusters[self.minimos[i]].append(example)

            # Obtener los nuevos centroides
            new_centroids = self.__compute_centroids()
            logging.info(f"Old centroids: {self.centroids}")
            logging.info(f"New centroids: {new_centroids}")
            logging.info(f"Clusters: {[len(x) for x in self.clusters]}")

            # Obtener la diferencia de los centroides antiguos con los nuevos
            diff = np.square(np.array(self.centroids) -
                             np.array(new_centroids))
            logging.info(f"Diff  : {diff}")

            # Mirar si todas las diferencias son menores que el CORTE, en caso
            # de que todos los centroides hayan variado menos que CORTE => parada = True
            parada = not np.any(diff > self.corte)
            if parada or self.it >= self.max_it:
                condition = False
            else:
                # Cambiar los centroides antiguos por los nuevos y limpiar los
                # CLUSTERS para volver a aplicar el algoritmo
                self.it += 1
                self.centroids = new_centroids
                for cluster in self.clusters:
                    cluster.clear()

        self.end = time.time()
        return

    # See: https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    def davies_bouldin_index(self, X, labels):
        # Convertir los centroides a un array de NumPy
        centroids = np.array(self.centroids)
        # Inicializar una matriz para almacenar las medidas de dispersión dentro de cada clúster
        dispersion = np.zeros(self.k)
        for i in range(self.k):
            # Calcular la dispersión dentro de cada clúster
            dispersion[i] = np.sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1)) / len(X[labels == i])

        centroid_distances = np.linalg.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        dispersions = dispersion + dispersion[:, np.newaxis]
        dispersions = dispersions * (np.ones(dispersions.shape) - np.eye(dispersions.shape[0]) )
        D = dispersions/(centroid_distances + np.eye(*centroid_distances.shape))

        return np.sum(np.max(D, axis=1)) / self.k

    def __compute_distance(self, examples, centroid) -> np.ndarray:
        match self.distance:
            case "euclidean": return np.linalg.norm(examples - centroid, axis=1)
            case "manhattan": return np.sum(np.abs(examples - centroid), axis=1)
            # Caso por defecto devolver distancia euclidea
            case _: return np.linalg.norm(examples - centroid, axis=1)

    def __compute_centroids(self) -> list[tuple[float, ...]]:
        centroids = []

        for cluster in self.clusters:
            values = [0.0 for _ in range(len(self.colums))]
            for example in cluster:
                values += example
            values = [x/len(cluster) for x in values]
            centroids.append(values)
        return centroids

    def __str__(self) -> str:
        return f"\n\n\n"+\
            f"RESULTADO\n" +\
            f"N ITs: {self.it+1}\n" +\
            f"Centroides: {self.centroids}\n" +\
            f"Clusters elemens: {[len(x) for x in self.clusters]}\n" +\
            f"N CLUSTERS: {len(self.clusters)}\n" +\
            f"DB indice: {self.davies_bouldin_index(self.df.values, self.minimos)}\n" +\
            f"SKDB indice: {davies_bouldin_score(self.df.values, self.minimos)}\n" +\
            f"Tiempo transcurrido: {self.end-self.start} segundos\n" 


def main():
    df = pd.read_csv("analisis.csv")
    kmeans = KMeans(df, colums=COLUMS, k=2, max_it=10, corte=0.01)
    kmeans.compute()
    print(kmeans)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s]: %(message)s", level=logging.INFO)
    main()
