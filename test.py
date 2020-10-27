import numpy as np
from pyswarm import pso
from geneticalgorithm import geneticalgorithm as ga
import matplotlib
import random

def mainpso():

    inputx = [0.1, 0.2, 0.3]
    #weights = [-0.5, 0.2, 0.6]
    #print(weights)   

    def func(w):
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        return w1 * inputx[0] + w2 * inputx[1] + w3 * inputx[2] - inputx[0]
    
    def con(w):
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        return [w1 * inputx[0] + w2 * inputx[1] + w3 * inputx[2] - inputx[0]]

    lb = [-1, -1, -1]
    ub = [1, 1, 1]
    xopt, fopt = pso(func, lb, ub, f_ieqcons=con)
    print(xopt)
    print(fopt)
    print('---')
    print(inputx[0])
    print(xopt[0] * inputx[0] + xopt[1] * inputx[1] + xopt[2] * inputx[2])

def mainga():
    def f(X):
        return np.sum(X)

    varbound=np.array([[0,10]]*3)

    model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

    model.run()

    convergence=model.report
    solution=model.output_dict

    print(convergence)
    print(solution)

if __name__ == '__main__':
    mainpso()
    #mainga()