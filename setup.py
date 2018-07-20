from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(['pyqentangle/interpolate_nocheck.pyx',
                             'pyqentangle/bipartite_reddenmat_nocheck.pyx',
                             'pyqentangle/bipartite_denmat.pyx'])
except ImportError:
    ext_modules = [Extension('_interpolate_nocheck',
                             sources=['pyqentangle/interpolate_nocheck.c']),
                   Extension('_bipartite_reddenmat_nocheck',
                             sources=['pyqentangle/bipartite_reddenmat_nocheck.c']),
                   Extension('_bipartite_denmat',
                             sources=['pyqentangle/bipartite_denmat.c'])]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyqentangle',
      version="1.0.2",
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
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['pyqentangle'],
      include_dirs=[np.get_include()],
      setup_requires=['Cython', 'numpy', ],
      install_requires=['numpy',],
      tests_require=['unittest2', 'numpy', 'scipy',],
      ext_modules=ext_modules,
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
