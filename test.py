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
    weightsFcm = np.zeros((n,n))

    def func(w):
        fresult = 0
        res = np.zeros(n)
        for i in range(0,n):
            result = 0
            wx = 0
            for j in range(0,m):
                wx = weightsAgg[i][j]
                weightsAgg[i][j] = w[i*m + j]
                #result+=w[i*m + j] * inputs[i][j]
                result += wx * inputs[i][j]
            aggr[i] = result
            for k in range(0,n):
                res[k] += aggr[i] * w[n*m + k*n + i]
        for l in range(0,n):

            fresult += (res[l] - inputs[l][m])**2
        #fresult += (res[1] - inputs[1][m])**2
        return fresult  

    lb = -np.ones(n*m + n*n)
    ub = np.ones(n*m + n*n)
    xopt, fopt = pso(func, lb, ub, maxiter = 10000, phig = 0.5, phip = 0.5, debug=False)
    
    #print(xopt)
    #print(fopt)
    #print(aggr)
    for t in range(0,n):
        for l in range(0,m):
            weightsAgg[t][l] = xopt[t*m + l]
    for r in range(0,n):
        for y in range(0,n):
            weightsFcm[r][y] = xopt[n*m + r*n + y]

    print("Aggregation weights:")    
    print(weightsAgg)
    print('---')
    print("Cognitive map weights:")
    print(weightsFcm)
    print('---')
    print("Results:")
    finres = np.zeros(n)
    for k in range(0, n):
        for l in range(0,n):
            finres[k] += xopt[n*m + k*n + l] * aggr[l]
    print(finres)

    return

if __name__ == '__main__':
    mainpso(3, 4)