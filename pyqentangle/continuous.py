from itertools import product

import numpy as np

import schmidt

disentangled_gaussian = lambda x1, x2: np.exp(-0.5*(x1*x1+x2*x2))/np.sqrt(np.pi)

class OutOfRangeException(Exception):
    def __init__(self, value):
        self.msg = "Out of range: "+str(value)

    def __str__(self):
        return repr(self.msg)

class UnequalLengthException(Exception):
    def __init__(self, array1, array2):
        self.msg = "Unequal length: "+str(len(array1))+" vs. "+str(len(array2))

    def __str__(self):
        return repr(self.msg)

def numerical_continuous_interpolation(xarray, yarray, x):
    if len(xarray) != len(yarray):
        raise UnequalLengthException(xarray, yarray)
    minx = np.min(xarray)
    maxx = np.max(xarray)
    if x==maxx:
        return yarray[-1]
    if not (x >= minx and x< maxx):
        raise OutOfRangeException(x)

    # assumed xarray sorted (for efficient run; reasonable assumption)
    # binary search
    left = 0
    right = len(xarray)-1
    idx = len(xarray) / 2
    while (idx!=0 and idx!=len(xarray)-1) and (not (x >= xarray[idx] and x < xarray[idx+1])):
        if x >= xarray[idx+1]:
            left = idx+1
        elif x < xarray[idx]:
            right = idx-1
        idx = (left + right) / 2

    # interpolation
    val = yarray[idx] + (yarray[idx+1]-yarray[idx])/(xarray[idx+1]-xarray[idx])*(x-xarray[idx])
    return val

def numerical_continuous_function(xarray, yarray):
    return lambda xs: np.array(map(lambda x: numerical_continuous_interpolation(xarray, yarray, x), xs))

def discretize_continuous_bipartitesys(fcn, x1_lo, x1_hi, x2_lo, x2_hi, nb_x1=100, nb_x2=100):
    x1 = np.linspace(x1_lo, x1_hi, nb_x1)
    x2 = np.linspace(x2_lo, x2_hi, nb_x2)
    tensor = np.zeros((len(x1), len(x2)))
    for i, j in product(*map(range, tensor.shape)):
        tensor[i, j] = fcn(x1[i], x2[j])
    return tensor

def continuous_schmidt_decomposition(fcn, x1_lo, x1_hi, x2_lo, x2_hi, nb_x1=100, nb_x2=100, keep=None):
    tensor = discretize_continuous_bipartitesys(fcn, x1_lo, x1_hi, x2_lo, x2_hi, nb_x1=nb_x1, nb_x2=nb_x2)
    decomposition = schmidt.schmidt_decomposition(tensor)

    if keep==None or keep>len(decomposition):
        keep = len(decomposition)

    x1array = np.linspace(x1_lo, x1_hi, nb_x1)
    x2array = np.linspace(x2_lo, x2_hi, nb_x2)
    dx1 = (x1_hi-x1_lo)/(nb_x1-1.)
    dx2 = (x2_hi-x2_lo)/(nb_x2-1.)

    renormalized_decomposition = []
    sumeigvals = np.sum(map(lambda dec: dec[0], decomposition))
    for i in range(keep):
        schmidt_weight, unnorm_modeA, unnorm_modeB = decomposition[i]
        sqnormA = np.linalg.norm(unnorm_modeA)**2*dx1
        sqnormB = np.linalg.norm(unnorm_modeB)**2*dx2
        renormalized_decomposition.append(
            (schmidt_weight/sumeigvals,
             numerical_continuous_function(x1array, unnorm_modeA/np.sqrt(sqnormA)),
             numerical_continuous_function(x2array, unnorm_modeB/np.sqrt(sqnormB))
             )
        )

    return renormalized_decomposition


