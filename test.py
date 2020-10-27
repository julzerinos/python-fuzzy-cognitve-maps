import numpy as np
from pyswarm import pso
from geneticalgorithm import geneticalgorithm as ga
import matplotlib
import random

def mainpso():

    inputx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #weights = [-0.5, 0.2, 0.6]
    #print(weights)
    weights = []
    output = []
    n = 3 #number of nodes

    for i in range(0,n):
        def func(w):
            result = 0
            for j in range(0,n): 
                result+=w[j] * inputx[j]
                #w1 = w[0]
                #w2 = w[1]
                #w3 = w[2]
            #return w1 * inputx[0 + i] + w2 * inputx[1 + i] + w3 * inputx[2 + i]
            return result
        
        def con(w):
            result = 0
            for j in range(0,n):
                result+=w[j] * inputx[j]
            #w1 = w[0]
            #w2 = w[1]
            #w3 = w[2]
            #return [w1 * inputx[0 + i] + w2 * inputx[1 + i] + w3 * inputx[2 + i] - inputx[1 + i]]
            return result - inputx[j - 1 + i]

        lb = -np.ones(n)
        ub = np.ones(n)
        xopt, fopt = pso(func, lb, ub, f_ieqcons=con, maxiter = 200)
        print(xopt)
        print(fopt)
        print('---')
        print(inputx[1 + i])
        res = 0
        for k in range(0, n):
            res += xopt[k] * inputx[k]
        #print(xopt[0] * inputx[0] + xopt[1] * inputx[1] + xopt[2] * inputx[2])
        print(res)

        weights.append(xopt)
        output.append(res)
    
    print(weights)
    print(output)


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