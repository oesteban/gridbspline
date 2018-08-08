# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Cython math extension '''
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs


cdef double c_cubic(double x) nogil:
    """
    Evaluate cubic spline at x
    """
    cdef:
        double x_t = fabs(x)

    if x_t >= 2.0:
        return 0.0
    if x_t < 1.0:
        return (4.0 - 6.0 * x_t * x_t + 3.0 * x_t * x_t * x_t) / 6.0
    elif x_t < 2.0:
        return (2 - x_t) ** 3 / 6.0

cdef double c_cubic_tensor(double x, double y, double z) nogil:
    cdef:
        bint inroi = (x * x + y * y + z * z) < 4.

    if not inroi:
        return 0.0
    return c_cubic(x) * c_cubic(y) * c_cubic(z)


def cubic(double x):
    """
    Evaluate the univariate cubic bspline at x

    Pure python implementation: ::

        def bspl(x):
            if x >= 2.0:
                return 0.0
            if x <= 1.0:
                return 2.0 / 3.0 - x**2 + 0.5 * x**3
            elif x <= 2.0:
                return (2 - x)**3 / 6.0
    """
    return(c_cubic(x))


def cubic_tensor(double x, double y, double z):
    """
    Evaluate BSpline tensor at xyz
    """
    return(c_cubic_tensor(x, y, z))
