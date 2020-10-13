
from .interpolate_nocheck import numerical_continuous_interpolation_nocheck_cython


def numerical_continuous_interpolation_nocheck(xarray, yarray, x):
    return numerical_continuous_interpolation_nocheck_cython(xarray, yarray, x)
