from setuptools import setup, Extension
#from Cython.Build import cythonize
import Cython.Build
import numpy as np

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyqentangle',
      version="0.22",
      description="Quantum Entanglement for Python",
      long_description="Schmidt decomposition for discrete and continuous bi-partite quantum systems",
      classifiers=[
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="quantum physics Schmidt decompostion entanglement",
      url="https://github.com/stephenhky/pyqentangle",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['pyqentangle'],
      include_dirs=[np.get_include()],
      setup_requires=['Cython',],
      install_requires=['numpy',],
      tests_require=['unittest2', 'numpy', 'scipy',],
      ext_modules=[Extension( 'interpolate_nocheck', ['pyqentangle/interpolate_nocheck.pyx']),],
      cmdclass={'build_ext': Cython.Build.build_ext},
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
