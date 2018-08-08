# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
GRIDBSPLINE
"""

from datetime import date
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright %d, Center for Reproducible Neuroscience, '
                 'Stanford University') % date.today().year
__credits__ = 'Oscar Esteban'
__license__ = 'Apache-2.0'
__status__ = 'Prototype'
__packagename__ = 'gridbspline'
__description__ = """\
N-Dimensional, gridded, and multivariate data interpolation using splines\
"""
__longdesc__ = """\
N-Dimensional, gridded, and multivariate data interpolation using splines. \
This package generalizes the interpolation code of P. Thévenaz \
(http://bigwww.epfl.ch/thevenaz/interpolation/) to Python.\
"""

__url__ = 'http://gridbspline.readthedocs.org/'
DOWNLOAD_URL = ('https://github.com/oesteban/gridbspline/archive/'
                '{}.tar.gz'.format(__version__))

SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

REQUIRES = [
    'versioneer',
]
LINKS_REQUIRES = []
TESTS_REQUIRES = []

EXTRA_REQUIRES = {
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme>=0.2.4',
        'imageio',
    ],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
