from setuptools import setup
from Cython.Build import cythonize
import numpy as np

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyqentangle',
      version="1.0.0",
      description="Quantum Entanglement for Python",
      long_description="Schmidt decomposition for discrete and continuous bi-partite quantum systems",
      classifiers=[
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
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
      ext_modules=cythonize('pyqentangle/interpolate_nocheck.pyx'),
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
