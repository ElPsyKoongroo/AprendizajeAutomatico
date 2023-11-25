import logging
import random
import time

# Librerias necesarias:
#   matplotlib  (pip install matplotlib)
#   numpy       (pip install numpy)
#   pandas      (pip install pandas)
#   sklearn     (pip install scikit-learn)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score

N_CLUSTERS = 2
NUMERO_IT = 100
CORTE = 0.01
COLUMS = [1, 3, 4]


#  ██████╗ ██╗   ██╗███╗   ██╗████████╗ ██████╗      ██╗
#  ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██╔═══██╗    ███║
#  ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║   ██║    ╚██║
#  ██╔═══╝ ██║   ██║██║╚██╗██║   ██║   ██║   ██║     ██║
#  ██║     ╚██████╔╝██║ ╚████║   ██║   ╚██████╔╝     ██║
#  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝   ╚═╝    ╚═════╝      ╚═╝
#
class KMeans:

    def __init__(self, df, colums, k=N_CLUSTERS, max_it=NUMERO_IT, corte=CORTE, distance="euclidean") -> None:
        self.__start = None
        self.__end = None
        self.minimos = None
        self.it = None

        self.colums = colums
        self.max_it = max_it
        self.corte = corte
        self.k = k
        self.distance = distance

        # Crear las listas vacias de centroides
        # y la lista de clusters vacios
        self.centroids: list[tuple[float, ...]] = [(0.0,) * k for _ in range(k)]
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

        # logging.info(f"Initial centroids: {self.centroids}")
        # logging.info(f"Initial cluster: {self.clusters}\n\n")

    def compute(self):
        condition = True
        self.it = 0
        self.__start = time.time()
        examples = np.array(self.df.values)
        filas, _ = examples.shape
        distancias = np.zeros((filas, self.k))
        while condition:
            logging.info(f"It {self.it}\n\n\n")

            # Calcular la primera columna de distancias euclidea con el centroide 0
            distancias[:, 0] = self.__compute_distance(
                examples, self.centroids[0])

            # Ir añadiendo las columnas de los otros centroides
            for i, centroid in enumerate(self.centroids[1:]):
                distancias[:, i + 1] = self.__compute_distance(
                    examples, centroid)
                # distancias = np.column_stack([distancias, distancias_i])

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
            for i, example in enumerate(examples):
                self.clusters[self.minimos[i]].append(example)

            # Obtener los nuevos centroides
            new_centroids = self.__compute_centroids()
            # logging.info(f"Old centroids: {self.centroids}")
            # logging.info(f"New centroids: {new_centroids}")
            # logging.info(f"Clusters: {[len(x) for x in self.clusters]}")

            # Obtener la diferencia de los centroides antiguos con los nuevos
            diff = np.square(np.array(self.centroids) - np.array(new_centroids))

            # logging.info(f"Diff  : {diff}")

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

        self.__end = time.time()
        return

    # See: https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    def davies_bouldin_index(self):
        x = self.df.values
        labels = self.minimos
        # Convertir los centroides a un array de NumPy
        centroids = np.array(self.centroids)
        # Inicializar una matriz para almacenar las medidas de dispersión dentro de cada clúster
        dispersion = np.zeros(self.k)
        for i in range(self.k):
            # Calcular la dispersión dentro de cada clúster
            dispersion[i] = np.sum(np.linalg.norm(
                x[labels == i] - centroids[i], axis=1)) / len(x[labels == i])

        centroid_distances = np.linalg.norm(
            centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        dispersions = dispersion + dispersion[:, np.newaxis]
        dispersions = dispersions * (np.ones(dispersions.shape) - np.eye(dispersions.shape[0]))
        d = dispersions / (centroid_distances + np.eye(*centroid_distances.shape))

        return np.sum(np.max(d, axis=1)) / self.k

    def inertia(self) -> np.float64:
        dispersion = np.zeros(self.k)
        for i in range(self.k):
            # Calcular la dispersión dentro de cada clúster
            dispersion[i] = np.sum(np.square(
                self.df.values[self.minimos == i] - self.centroids[i]))

        return np.sum(dispersion)

    def time(self) -> float:
        return self.__end - self.__start

    def __compute_distance(self, examples, centroid) -> np.ndarray:
        match self.distance:
            case "euclidean":
                return np.linalg.norm(examples - centroid, axis=1)
            case "manhattan":
                return np.sum(np.abs(examples - centroid), axis=1)
            # Caso por defecto devolver distancia euclidea
            case _:
                return np.linalg.norm(examples - centroid, axis=1)

    def __compute_centroids(self) -> list[tuple[float, ...]]:
        centroids = []

        for i, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                logging.error(f"Cluster vacio k={self.k}, it={self.it}")
                logging.warning(f"No se calculará un nuevo centroide, se usará el antiguo")
                centroids.append(self.centroids[i])
            else:
                s = np.sum(cluster, axis=0)
                new_centroid = s / len(cluster)
                centroids.append(new_centroid)
        return centroids

    def __str__(self) -> str:
        return f"\n\n\n" + \
            f"RESULTADO\n" + \
            f"N ITs: {self.it + 1}\n" + \
            f"Centroides: {self.centroids}\n" + \
            f"Clusters elemens: {[len(x) for x in self.clusters]}\n" + \
            f"N CLUSTERS: {len(self.clusters)}\n" + \
            f"DB   indice: {self.davies_bouldin_index()}\n" + \
            f"SKDB indice: {davies_bouldin_score(self.df.values, self.minimos)}\n" + \
            f"Tiempo transcurrido: {self.time()} segundos\n"


def main():
    results = []
    df = pd.read_csv("analisis.csv")

    #  ██████╗ ██╗   ██╗███╗   ██╗████████╗ ██████╗     ██████╗
    #  ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██╔═══██╗    ╚════██╗
    #  ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║   ██║     █████╔╝
    #  ██╔═══╝ ██║   ██║██║╚██╗██║   ██║   ██║   ██║    ██╔═══╝
    #  ██║     ╚██████╔╝██║ ╚████║   ██║   ╚██████╔╝    ███████╗
    #  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝   ╚═╝    ╚═════╝     ╚══════╝
    #
    results_df = pd.DataFrame(columns=["K", "Daivis-Bouldin"])
    for k in range(2, 60):
        kmeans = KMeans(df, colums=COLUMS, k=k, max_it=20, corte=0.01, distance="euclidean")
        kmeans.compute()
        results.append((k, kmeans.inertia()))
        results_df.loc[len(results_df), :] = [k, kmeans.davies_bouldin_index()]

    print(results_df.to_string(index=False))

    #  ██████╗ ██╗   ██╗███╗   ██╗████████╗ ██████╗     ██████╗
    #  ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██╔═══██╗    ╚════██╗
    #  ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║   ██║     █████╔╝
    #  ██╔═══╝ ██║   ██║██║╚██╗██║   ██║   ██║   ██║     ╚═══██╗
    #  ██║     ╚██████╔╝██║ ╚████║   ██║   ╚██████╔╝    ██████╔╝
    #  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝   ╚═╝    ╚═════╝     ╚═════╝

    df_ordenado = results_df.sort_values(by="Daivis-Bouldin", ascending=False)
    print("\n\nLos 5 mejores K segun el Davies-Bouldin index son: ")
    print(df_ordenado[:5].to_string(index=False))

    x, y = zip(*results)
    plt.plot(x, y, marker='o', linestyle='-')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    #  ██████╗ ██╗   ██╗███╗   ██╗████████╗ ██████╗     ██╗  ██╗
    #  ██╔══██╗██║   ██║████╗  ██║╚══██╔══╝██╔═══██╗    ██║  ██║
    #  ██████╔╝██║   ██║██╔██╗ ██║   ██║   ██║   ██║    ███████║
    #  ██╔═══╝ ██║   ██║██║╚██╗██║   ██║   ██║   ██║    ╚════██║
    #  ██║     ╚██████╔╝██║ ╚████║   ██║   ╚██████╔╝         ██║
    #  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝   ╚═╝    ╚═════╝          ╚═╝
    #

    kmeans = KMeans(df, colums=COLUMS, k=5, max_it=20, corte=0.01, distance="euclidean")
    kmeans.compute()

    colores = ["red", "green", "blue", "yellow", "pink"]
    for i, clase in enumerate(set(kmeans.minimos)):
        values = kmeans.df.values[kmeans.minimos == clase, :]
        ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=colores[i], label=f"Clase {i}")

    for i, centroid in enumerate(kmeans.centroids):
        ax.scatter(*centroid, c=colores[i], marker='*', s=200, label=f"Centroide clase {i}")

    ax.set_xlabel(df.columns[kmeans.colums[0]])
    ax.set_ylabel(df.columns[kmeans.colums[1]])
    ax.set_zlabel(df.columns[kmeans.colums[2]])
    ax.set_title(f'Clusters = {kmeans.k}')

    # La leyenda suele salir un poco donde le da la gana
    # Aumentar el tamaño de la ventana para que se pueda ver
    # bien y no aparezca en medio del grafico
    ax.legend(loc="lower right", fontsize=12, bbox_to_anchor=(-0.2, 0.0))

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s]: %(message)s", level=logging.WARN)
    main()
