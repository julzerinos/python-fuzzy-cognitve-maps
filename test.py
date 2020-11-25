import numpy as np
from pyswarm import pso
import matplotlib
import random
import math
#pymoo
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.so_de import DE
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.optimize import minimize
#scipy
from scipy.optimize import minimize as minsci
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
#lmfit
from lmfit import Parameters
from lmfit import Minimizer

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
        for i in range(0,n):            result = 0
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

def mainmoo():
    
    inputx = [0.1, 0.2, 0.3, 0.4, 0.5]

    class MyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=3, n_constr=1, xl=-np.ones(3), xu=np.ones(3), elementwise_evaluation=True)
        def _evaluate(self, x, out, *args, **kwargs):
            f1 = x[0]*inputx[0] + x[1]*inputx[1] + x[2]*inputx[2]
            g1 = -(x[0]*inputx[0] + x[1]*inputx[1] + x[2]*inputx[2] - inputx[3])
            out["F"] = f1
            out["G"] = g1
    
    elementwise_problem = MyProblem()

#to działa
    #algorithm = GA(pop_size=100, eliminate_duplicates=True)
#
    #val = [0.47013193, 0.51047106, 0.8363237 ]
    val = [0.5, 0.5, 0.5]
    pop = np.ndarray(shape=(1,3), buffer=np.array(val))
    print(pop)

#to nie działa, jak chcemy
#sampling to tylko zbiór wartości, z których może korzystać, nie są mutowane
#można dać kilka istniejących rozwiązań i wtedy je przemiesza
    #algorithm = DE(pop_size=100, sampling=pop, variant="DE/rand/1/bin", F=0.3, CR=0.5)


    algorithm = CMAES(x0=pop, restarts = 2, tolfun=1e-4, tolx=1e-4, restart_from_best = True, bipop=True)
#cmaes jest jakieś dzikie, trzeba ograniczyć iteracje i verbose=False
    
    res = minimize(elementwise_problem, algorithm, ('n_iter', 5000), seed=1, verbose=False)
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    #print(res.X[0][0] * inputx[0] + res.X[0][1] * inputx[1] + res.X[0][2] * inputx[2])
    #print(res.X[0][1])

    return

def mainscipy():
    
    n=3
    m=4

    inputx = [0.1, 0.2, 0.3, 0.4, 0.5]
    inputy = [0.2, 0.4, 0.6, 0.8, 1]
    inputz = [0.3, 0.6, 0.9, 1.2, 1.5]
    inputs = []
    inputs.append(inputx)
    inputs.append(inputy)
    inputs.append(inputz)

    aggr = np.zeros(n)
    weightsAgg = np.zeros((n,m))
    weightsFcm = np.zeros((n,n))

    #def func(x):
    #    return x[0]*inputx[0] + x[1]*inputx[1] + x[2]*inputx[2]

    def func(w):
        fresult = 0
        res = np.zeros(n)
        for i in range(0,n):
            result = 0
            wx = 0
            for j in range(0,m):
                #wx = weightsAgg[i][j]
                #weightsAgg[i][j] = w[i*m + j]
                result+=w[i*m + j] * inputs[i][j]
                #result += wx * inputs[i][j]
            aggr[i] = result
            for k in range(0,n):
                res[k] += aggr[i] * w[n*m + k*n + i]
        for l in range(0,n):
            fresult += res[l]
        #fresult += (res[1] - inputs[1][m])**2
        return fresult 

    #def cons(x):
    #    return [x[0]*inputx[0] + x[1]*inputx[1] + x[2]*inputx[2] - inputx[3]]

    def cons(w):
        fresult = 0
        res = np.zeros(n)
        for i in range(0,n):
            result = 0
            wx = 0
            for j in range(0,m):
                #wx = weightsAgg[i][j]
                #weightsAgg[i][j] = w[i*m + j]
                result+=w[i*m + j] * inputs[i][j]
                #result += wx * inputs[i][j]
            aggr[i] = result
            for k in range(0,n):
                res[k] += aggr[i] * w[n*m + k*n + i]
        for l in range(0,n):
            fresult += (res[l] - inputs[l][m])**2
        #fresult += (res[1] - inputs[1][m])**2
        return fresult 

    #x0 = np.array([0.5, 0.5, 0.5])
    #x0 = np.array([0.45512582, 0.65955673, 0.74192024])

    x0 = np.random.rand(21)
    
    #lb = -np.ones(3)
    #ub = np.ones(3)

    lb = -np.ones(n*m + n*n)
    ub = np.ones(n*m + n*n)

    bnds = Bounds(lb, ub)
    nonlinc = NonlinearConstraint(cons, 0, 0)

    #res = minsci(func, x0, method='Powell', bounds=bnds, options={'disp': True})
    res = minsci(func, x0, method='trust-constr', constraints=nonlinc, options={'disp': True}, bounds=bnds)
    #res = minsci(func, x0, method='trust-constr', options={'disp': True}, bounds=bnds)

    #print(res.x)
    #print('Result:')
    #print(res.x[0] * inputx[0] + res.x[1] * inputx[1] + res.x[2] * inputx[2])

    for t in range(0,n):
        for l in range(0,m):
            weightsAgg[t][l] = res.x[t*m + l]
    for r in range(0,n):
        for y in range(0,n):
            weightsFcm[r][y] = res.x[n*m + r*n + y]

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
            finres[k] += res.x[n*m + k*n + l] * aggr[l]
    print(finres)

    return

def mainlmfit():

    #for i in range(1,5):
    #    print('x' + str(i))

    inputx = [0.1, 0.2, 0.3, 0.4, 0.5]

    params = Parameters()
    params.add('w1', value=0.5, min=-1, max=1)
    params.add('w2', value=0.5, min=-1, max=1)
    params.add('w3', value=0.5, min=-1, max=1)

    def func(par):
        return math.sqrt((par['w1']*inputx[0] + par['w2']*inputx[1] + par['w3']*inputx[2] - inputx[3])**2)

    fitter = Minimizer(func, params)
    result = fitter.minimize(method='nelder')
    result.params.pretty_print()
    
    #print(result.params)
    res = result.params
    print('Result:')
    print(res['w1']*inputx[0] + res['w2']*inputx[1] + res['w3']*inputx[2])

    return


def bar():
    return 2, 1

def foo():
    return 3, bar()

a, b = foo()






if __name__ == '__main__':
    #mainpso(3, 4)
    #mainmoo()
    #mainscipy()
    mainlmfit()