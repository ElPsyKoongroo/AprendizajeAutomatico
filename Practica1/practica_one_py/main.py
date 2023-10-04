import csv
import numpy as np
import pandas as pd

LEARNING_RATE = 0.001
ITERS = 1000

# En porcentaje
BATCH = 100

def hypotesis(thetas, X) -> float :
    h = 0.0
    for i,entry in enumerate(X):
        h += entry*thetas[i+1]
    h+=thetas[0]

    return h
    

#def cost_function():
#    return (1/2)


    '''
    theta0 = (1/m) * np.sum(hypothesis(thetas, X) - 1)
    theta1 = (1/m) * np.sum((hypothesis(thetas, X) - 1)*X)

    theta0 -= learning_rate*gradient_theta0
    theta1 -= learning_rate*gradient_theta1
    '''


def main():
    data = pd.read_csv("regresion_1.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    
    THETAS = [0.0]*X.ndim

    for _ in range(0, ITERS):
        errors = [0.0]*X.size
        for i, entry in enumerate(X):
            errors[i] = hypotesis(THETAS, entry)-y[i]

        for i, theta in enumerate(THETAS):
            sum = 0
            for row,instance in enumerate(X):
                sum += errors[row] * instance[i]
            THETAS[i] = theta - LEARNING_RATE*sum

    print(THETAS)


    

if __name__ == "__main__":
    main()
