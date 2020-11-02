import numpy as np
from pyswarm import pso
import matplotlib
import random

def mainpso(n, m):

    #n = number of nodes
    #m = size of window

    #-----
    #input, matrix n * (m + 1)
    inputx = [0.1, 0.2, 0.3, 0.4, 0.5]
    inputy = [0.2, 0.4, 0.6, 0.8, 1]
    inputz = [0.3, 0.6, 0.9, 1.2, 1.5]
    inputs = []
    inputs.append(inputx)
    inputs.append(inputy)
    inputs.append(inputz)
    #-----
    
    aggr = np.zeros(n)
    weightsAgg = np.zeros((n,m))
    weightsFcm = np.zeros(n)

    def func(w):
        res = 0
        for i in range(0,n):
            result = 0
            wx = 0
            for j in range(0,m):
                wx = weightsAgg[i][j]
                weightsAgg[i][j] = w[i*m + j]
                #result+=w[i*m + j] * inputs[i][j]
                result += wx * inputs[i][j]
            aggr[i] = result
            res += aggr[i] * w[n*m + i]
        return (res - inputs[0][m])**2    

    lb = -np.ones(n*m + n)
    ub = np.ones(n*m + n)
    xopt, fopt = pso(func, lb, ub, maxiter = 2000, phig = 0.1, phip = 0.1, debug=False)
    
    #print(xopt)
    #print(fopt)
    #print(aggr)
    ind = 0
    for t in range(0,n):
        for l in range(0,m):
            weightsAgg[t][l] = xopt[t*m + l]
    for t in range(n*m, n*m + n):
        weightsFcm[ind] = xopt[t]
        ind += 1

    print("Aggregation weights:")    
    print(weightsAgg)
    print('---')
    print("Cognitive map weights:")
    print(weightsFcm)
    print('---')
    print("Result:")
    finres = 0
    for k in range(0, n):
        finres += xopt[n*m + k] * aggr[k]
    print(finres)

    return

if __name__ == '__main__':
    mainpso(3, 4)