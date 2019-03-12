
import numpy as np
cimport numpy as np


def numerical_continuous_interpolation_nocheck_cython(np.ndarray xarray, np.ndarray yarray, double x):
    cdef int left = 0
    cdef int right = len(xarray) - 1
    cdef int idx = len(xarray) / 2
    while (idx != 0 and idx != len(xarray) - 1) and (not (x >= xarray[idx] and x < xarray[idx + 1])):
        if x >= xarray[idx + 1]:
            left = idx + 1
        elif x < xarray[idx]:
            right = idx - 1
        idx = (left + right) / 2

    # interpolation
    cdef double val = yarray[idx] + (yarray[idx + 1] - yarray[idx]) / (xarray[idx + 1] - xarray[idx]) * (x - xarray[idx])
    return val
