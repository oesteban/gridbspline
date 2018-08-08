# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

"""
from sys import float_info
import numpy as np
from .maths import cubic

DBL_EPSILON = float_info.epsilon

POLES = {
    2: [np.sqrt(8.0) - 3],
    3: [np.sqrt(3.0) - 2],
    4: [
        np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
        np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0],
    5: [
        np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) +
        np.sqrt(105.0 / 4.0) - 13.0 / 2.0,
        np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) -
        np.sqrt(105.0 / 4.0) - 13.0 / 2.0],
    6: [
        -0.48829458930304475513011803888378906211227916123938,
        -0.081679271076237512597937765737059080653379610398148,
        -0.0014141518083258177510872439765585925278641690553467],
    7: [
        -0.53528043079643816554240378168164607183392315234269,
        -0.12255461519232669051527226435935734360548654942730,
        -0.0091486948096082769285930216516478534156925639545994],
    8: [
        -0.57468690924876543053013930412874542429066157804125,
        -0.16303526929728093524055189686073705223476814550830,
        -0.023632294694844850023403919296361320612665920854629,
        -0.00015382131064169091173935253018402160762964054070043],
    9: [
        -0.60799738916862577900772082395428976943963471853991,
        -0.20175052019315323879606468505597043468089886575747,
        -0.043222608540481752133321142979429688265852380231497,
        -0.0021213069031808184203048965578486234220548560988624]
}


class BsplineNDInterpolator(object):
    """
    Interpolation on a regular grid of arbitrary dimensions
    The data must be defined on a regular grid.

    Parameters
    ----------
    data : array_like, shape (X1, X2, ..., Xn, V)
        A regular data array, where X1, ..., Xn are the n dimensions
        of the grid, and V the number of components of a multivariate
        value at each location.
    order : int, optional
        B-Spline order (default is 3)
    off_bounds : str
        Strategy when interpolating points outsied the grid bounds.
        Either `'mirror'`, `'value'` or `'error'`, default is `'mirror'`.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Example
    -------

    >>> import imageio
    >>> import numpy as np
    >>> img = imageio.imread(
    ...     'https://farm9.staticflickr.com/8674/16504233985_9f1060624e_m_d.jpg')
    >>> interpolator = BsplineNDInterpolator(img)
    >>> mat = np.eye(3)
    >>> mat[:2, 2] = [-13.5, 24.2]
    >>> indexes = (list(range(img.shape[0])), list(range(img.shape[1])))
    >>> ndindex = np.moveaxis(np.array(np.meshgrid(
    ...     *indexes, indexing='ij')), 0, -1).reshape(-1, 2)
    >>> mapped = np.dot(mat, np.hstack(
    ...     (ndindex.astype(float), np.ones((ndindex.shape[0], 1)))).T)[:2, :].T
    >>> newdata = np.array(list(interpolator(mapped))).reshape(img.shape)
    >>> imageio.imwrite('newdata.jpg', np.clip(newdata, 0, 255).astype('uint8'))


    References
    ----------
    .. [1] P. Thévenaz, T. Blu, M. Unser, "Interpolation Revisited," IEEE Transactions
           on Medical Imaging, vol. 19, no. 7, pp. 739-758, July 2000.
    """
    slots = ['_order', '_off_bounds', '_coeffs', '_fill_value', 'ndim', 'ncomp', 'shape']

    def __init__(self, data, order=3, off_bounds='mirror', fill_value=0.0):
        data = np.array(data, dtype='float32')
        self._order = order
        self._off_bounds = off_bounds
        self._fill_value = fill_value
        self._poles = np.array(POLES[order])
        self.shape = data.shape[:-1]
        self.ndim = len(self.shape)
        self.ncomp = data.shape[-1]
        self.ncoeff = np.prod(self.shape)

        # Do not modify original data
        # coefficients are updated in-place by filtering
        self._coeffs = data.copy()

        for dim in reversed(range(self.ndim)):
            coeffs = np.moveaxis(
                np.moveaxis(self._coeffs, dim, 0).reshape(self.shape[dim], -1, self.ncomp),
                0, 1)
            for line in coeffs:
                line = _samples_to_coeffs(line, self._poles)

    def __call__(self, coords):
        """
        Interpolation at coordinates
        Parameters
        ----------
        coords : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        """

        if self._order != 3:
            raise NotImplementedError

        for xi in coords:
            yield self._interpolate(xi)

    def _interpolate(self, xi):
        """Evaluates the interpolated value for position xi

        Calculates the bspline weights corresponding to the samples
        around xi and evaluates the interpolated value

        Parameters
        ----------

        xi : array_like of shape (ndim,)
            The position at which the image is interpolated

        """
        if self._order != 3:
            raise NotImplementedError

        indexes = []
        offset = 0.0 if self._order & 1 else 0.5
        for dim in range(self.ndim):
            first = int(np.floor(xi[dim] + offset) - self._order // 2)
            indexes.append(list(range(first, first + self._order + 1)))

        ndindex = np.moveaxis(
            np.array(np.meshgrid(*indexes, indexing='ij')), 0, -1).reshape(
            -1, self.ndim)

        vbspl = np.vectorize(cubic)
        weights = np.prod(vbspl(ndindex - xi), axis=-1)
        ndindex = [tuple(v) for v in ndindex]

        zero = np.zeros(self.ndim)
        shape = np.array(self.shape)
        coeffs = []
        for ijk in ndindex:
            offbounds = (zero > ijk) | (shape <= ijk)
            if np.any(offbounds):
                # Deal with offbounds samples
                if self._off_bounds == 'constant':
                    coeffs.append([0.0] * self.ncomp)
                    continue
                ijk = np.array(ijk, dtype=int)
                ijk[ijk < 0] *= -1
                ijk[ijk >= shape] = (2 * shape[ijk >= shape] - ijk[ijk >= shape] - 1).astype(int)
                ijk = tuple(ijk.tolist())

            coeffs.append(self._coeffs[ijk])
        return weights.dot(np.array(coeffs, dtype=float))


def _samples_to_coeffs(line, poles, tol=DBL_EPSILON):
    """Filter a 1D series to obtain the corresponding coefficients

    BSplines are separable, and therefore dimensions can be filtered
    sequentially. Processing of "lines" is independent (i.e. parallelizable)

    Parameters
    ----------

    line : array_like of shape (N, C)
        N is the number of samples along the processed axis
        and C is the number of components of the data.
    poles : list of float
        Poles corresponding to the B-Spline basis selected

    tol : float
        A tolerance value extending filtering to infinite

    """
    # Compute the overall gain and apply
    gain = np.prod((1 - poles) * (1 - 1. / poles))
    line *= gain

    for p in poles:
        # causal initialization
        line[0] = _causal_c0(line, p, tol)
        # causal recursion
        for n in range(1, len(line)):
            line[n] += p * line[n - 1]
        # anticausal initialization
        line[-1] = _anticausal_cn(line, p)
        # anticausal recursion
        for n in reversed(range(0, len(line) - 1)):
            line[n] = p * (line[n + 1] - line[n])

    return line


def _causal_c0(line, z, tol=DBL_EPSILON):
    """Calculate the first coefficient of the causal filter"""
    length = len(line)
    horiz = length
    if tol > 0:
        horiz = (np.ceil(np.log(tol)) / np.log(np.abs(z))).astype(int)

    zn = float(z)
    if horiz < length:
        # Accelerated loop
        csum = line[0]
        for n in range(horiz):
            csum += zn * line[n]
            zn *= z
        return csum

    # Full loop
    iz = 1.0 / z
    z2n = z ** length
    csum = line[0] + z2n * line[-1]
    z2n *= z2n * iz
    for n in range(length):
        csum += (zn + z2n) * line[n]
        zn *= z
        z2n *= iz

    return csum / (1.0 - zn ** 2)


def _anticausal_cn(line, z):
    """Calculate the last coefficient of the anticausal filter"""
    return (z / (z * z - 1.0)) * (z * line[-2] + line[-1])
