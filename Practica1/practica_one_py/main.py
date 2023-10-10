import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#       _______. _______ .______        _______  __    ______          
#      /       ||   ____||   _  \      /  _____||  |  /  __  \         
#     |   (----`|  |__   |  |_)  |    |  |  __  |  | |  |  |  |        
#      \   \    |   __|  |      /     |  | |_ | |  | |  |  |  |        
#  .----)   |   |  |____ |  |\  \----.|  |__| | |  | |  `--'  |        
#  |_______/    |_______|| _| `._____| \______| |__|  \______/         
#                                                                      
#    _______      ___      .______        ______  __       ___         
#   /  _____|    /   \     |   _  \      /      ||  |     /   \        
#  |  |  __     /  ^  \    |  |_)  |    |  ,----'|  |    /  ^  \       
#  |  | |_ |   /  /_\  \   |      /     |  |     |  |   /  /_\  \      
#  |  |__| |  /  _____  \  |  |\  \----.|  `----.|  |  /  _____  \     
#   \______| /__/     \__\ | _| `._____| \______||__| /__/     \__\    
#                                                                      
#  .___  ___.      ___        ______  __       ___           _______.  
#  |   \/   |     /   \      /      ||  |     /   \         /       |  
#  |  \  /  |    /  ^  \    |  ,----'|  |    /  ^  \       |   (----`  
#  |  |\/|  |   /  /_\  \   |  |     |  |   /  /_\  \       \   \      
#  |  |  |  |  /  _____  \  |  `----.|  |  /  _____  \  .----)   |     
#  |__|  |__| /__/     \__\  \______||__| /__/     \__\ |_______/      
#                                                                      


LEARNING_RATE = 0.01
ITERS = 1_000

# En porcentaje
BATCH = 50

def representar(x, y):
    plt.scatter(x, y)
    plt.show()


#               _         _____    ____    _____    _____   _______   __  __    ____     _____ 
#       /\     | |       / ____|  / __ \  |  __ \  |_   _| |__   __| |  \/  |  / __ \   / ____|
#      /  \    | |      | |  __  | |  | | | |__) |   | |      | |    | \  / | | |  | | | (___  
#     / /\ \   | |      | | |_ | | |  | | |  _  /    | |      | |    | |\/| | | |  | |  \___ \ 
#    / ____ \  | |____  | |__| | | |__| | | | \ \   _| |_     | |    | |  | | | |__| |  ____) |
#   /_/    \_\ |______|  \_____|  \____/  |_|  \_\ |_____|    |_|    |_|  |_|  \____/  |_____/ 
#                                                                                              

def hypotesis(thetas: np.ndarray[np.float64], entry: np.ndarray[np.float64]) -> float :
    # Multiplicacion de matrices + suma de sus elementos
    return np.dot(thetas, entry);

def descenso_gradiente(X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    (ROWS, COLUMS) = X.shape
    thetas = np.array([0.0]*COLUMS, dtype=np.float64)

    evolucion_error = []
    square = lambda x: pow(x,2)
    errors = np.array([0.0]*ROWS, dtype=np.float64)
    for _ in range(0, ITERS):
        for i, entry in enumerate(X):
            errors[i] = hypotesis(thetas, entry)-y[i]
            
        for i, _ in enumerate(thetas):
            gradiente = np.dot(X[:, i].T, errors)
            thetas[i] -= LEARNING_RATE/ROWS*gradiente

    print("[Descenso por gradiente] ", thetas)
    return thetas

def descenso_gradiente_estocastico(fX: np.ndarray[np.float64], fy: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # Seleccionar el BATCH a calcular mediante porcentaje
    i = fX.shape[0] * BATCH // 100
    X = fX[:i, :]
    y = fy[:i]
    
    (ROWS, COLUMS) = X.shape
    thetas = np.array([0.0]*COLUMS, dtype=np.float64)


    errors = np.array([0.0]*ROWS, dtype=np.float64)
    
    for _ in range(0, ITERS):
        for i, entry in enumerate(X):
            errors[i] = hypotesis(thetas, entry)-y[i]
            
        for i, _ in enumerate(thetas):
            gradiente = np.dot(X[:, i].T, errors)
            thetas[i] -= LEARNING_RATE/ROWS*gradiente
    print("[Descenso por gradiente estocastico] ", thetas)
    return thetas

def ecuacion_normal(X: np.ndarray[np.float64], y: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    thetas = np.linalg.inv(X.T @ X) @ X.T @ y;
    print("[Ecuacion normal] ", thetas)
    return thetas

def mide_tiempo(funcion: callable, args: any) -> any:
    import time

    inicio = time.perf_counter()
    res = funcion(*args)
    fin = time.perf_counter()
    nombre = funcion.__name__
    print(f"[TIEMPO DE: {nombre}] -> {(fin - inicio)*1000:.2f}ms")
    return res

def prediccion(thetas: np.ndarray[np.float64], x: float) -> float:
    pred = 0.0
    for theta in thetas[1:]:
        pred += theta*x
    return pred+thetas[0]


#  __/\\\\____________/\\\\_        _____/\\\\\\\\\____        __/\\\\\\\\\\\_        __/\\\\\_____/\\\_        
#   _\/\\\\\\________/\\\\\\_        ___/\\\\\\\\\\\\\__        _\/////\\\///__        _\/\\\\\\___\/\\\_       
#    _\/\\\//\\\____/\\\//\\\_        __/\\\/////////\\\_        _____\/\\\_____        _\/\\\/\\\__\/\\\_      
#     _\/\\\\///\\\/\\\/_\/\\\_        _\/\\\_______\/\\\_        _____\/\\\_____        _\/\\\//\\\_\/\\\_     
#      _\/\\\__\///\\\/___\/\\\_        _\/\\\\\\\\\\\\\\\_        _____\/\\\_____        _\/\\\\//\\\\/\\\_    
#       _\/\\\____\///_____\/\\\_        _\/\\\/////////\\\_        _____\/\\\_____        _\/\\\_\//\\\/\\\_   
#        _\/\\\_____________\/\\\_        _\/\\\_______\/\\\_        _____\/\\\_____        _\/\\\__\//\\\\\\_  
#         _\/\\\_____________\/\\\_        _\/\\\_______\/\\\_        __/\\\\\\\\\\\_        _\/\\\___\//\\\\\_ 
#          _\///______________\///__        _\///________\///__        _\///////////__        _\///_____\/////__

def main():
    data = pd.read_csv("regresion_1.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    pad_width = ((0, 0), (1, 0)) 
    X = np.pad(X, pad_width, mode='constant', constant_values=1)
    
    thetas = mide_tiempo(descenso_gradiente, (X, y))
    print("\n\n")
    mide_tiempo(descenso_gradiente_estocastico,(X,y))
    print("\n\n")
    mide_tiempo(ecuacion_normal, (X, y))

    p = prediccion(thetas, 9.5)
    print(p)
    
  
if __name__ == "__main__":
    main()
