#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" GRIDBSPLINE setup script """


def main():
    """ Install entry-point """
    import os
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    from numpy import get_include

    from gridbspline.__about__ import (  # noqa
        __version__,
        __author__,
        __email__,
        __maintainer__,
        __copyright__,
        __credits__,
        __license__,
        __status__,
        __description__,
        __longdesc__,
        __url__,
        __packagename__,
        DOWNLOAD_URL,
        CLASSIFIERS,
        REQUIRES,
        SETUP_REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
    )

    pkg_data = {
        __packagename__: []
    }

    version = None
    cmdclass = {}
    root_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(os.path.join(root_dir, __packagename__, 'VERSION')):
        with open(os.path.join(root_dir, __packagename__, 'VERSION')) as vfile:
            version = vfile.readline().strip()
        pkg_data[__packagename__].insert(0, 'VERSION')

    if version is None:
        import versioneer
        version = versioneer.get_version()
        cmdclass = versioneer.get_cmdclass()

    extensions = [Extension(
        "gridbspline.maths",
        ["gridbspline/maths.pyx"],
        include_dirs=[get_include(), "/usr/local/include/"],
        library_dirs=["/usr/lib/"]),
    ]

    setup(
        name=__packagename__,
        version=version,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        license=__license__,
        url=__url__,
        maintainer_email=__email__,
        classifiers=CLASSIFIERS,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        setup_requires=SETUP_REQUIRES,
        install_requires=REQUIRES,
        dependency_links=LINKS_REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        package_data=pkg_data,
        entry_points={'console_scripts': []},
        packages=find_packages(exclude=('tests',)),
        zip_safe=False,
        ext_modules=extensions,
        cmdclass=cmdclass,
    )


if __name__ == '__main__':
    main()
