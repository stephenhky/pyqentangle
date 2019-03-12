from setuptools import setup, Extension
import numpy as np

# reference: https://stackoverflow.com/questions/46784964/create-package-with-cython-so-users-can-install-it-without-having-cython-already
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(['pyqentangle_Hosseinberg/interpolate_nocheck.pyx',
                             'pyqentangle_Hosseinberg/bipartite_reddenmat_nocheck.pyx',
                             'pyqentangle_Hosseinberg/bipartite_denmat.pyx',
                             'pyqentangle_Hosseinberg/negativity_utils.pyx'])
except ImportError:
    ext_modules = [Extension('pyqentangle_Hosseinberg.interpolate_nocheck',
                             sources=['pyqentangle_Hosseinberg/interpolate_nocheck.c']),
                   Extension('pyqentangle_Hosseinberg.bipartite_reddenmat_nocheck',
                             sources=['pyqentangle_Hosseinberg/bipartite_reddenmat_nocheck.c']),
                   Extension('pyqentangle_Hosseinberg.bipartite_denmat',
                             sources=['pyqentangle_Hosseinberg/bipartite_denmat.c']),
                   Extension('pyqentangle_Hosseinberg.negativity_utils',
                             sources=['pyqentangle_Hosseinberg/negativity_utils.c'])]


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pyqentangle_Hosseinberg',
      version="2.0.0",
      description="Quantum Entanglement for Python",
      long_description="Schmidt decomposition for discrete and continuous bi-partite quantum systems",
      classifiers=[
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Chemistry",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Cython",
          "Programming Language :: C",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Intended Audience :: Education"
      ],
      keywords="quantum physics Schmidt decompostion entanglement",
      url="https://github.com/stephenhky/pyqentangle",
      author="Hossein",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['pyqentangle_Hosseinberg'],
      include_dirs=[np.get_include()],
      setup_requires=['Cython', 'numpy', ],
      install_requires=['numpy',],
      tests_require=['unittest2', 'numpy', 'scipy',],
      ext_modules=ext_modules,
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
