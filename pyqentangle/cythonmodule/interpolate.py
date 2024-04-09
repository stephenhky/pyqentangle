
from .interpolate_nocheck import numerical_continuous_interpolation_nocheck_cython


def numerical_continuous_interpolation_nocheck(xarray, yarray, x):
    return numerical_continuous_interpolation_nocheck_cython(xarray, yarray, x)


def interpolate(xarray, yarray, x):
    left = 0
    right = len(xarray) - 1
    idx = right // 2
    while (idx != 0 and idx != len(xarray) - 1) and (not (x >= xarray[idx] and x < xarray[idx + 1])):
        if x >= xarray[idx + 1]:
            left = idx + 1
        elif x < xarray[idx]:
            right = idx - 1
        idx = (left + right) // 2

    return yarray[idx] + (yarray[idx + 1] - yarray[idx]) / (xarray[idx + 1] - xarray[idx]) * (x - xarray[idx])
