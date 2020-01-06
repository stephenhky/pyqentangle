from setuptools import setup, Extension
import numpy as np

# reference: https://stackoverflow.com/questions/46784964/create-package-with-cython-so-users-can-install-it-without-having-cython-already
try:
    from Cython.Build import cythonize
    ext_modules = cythonize(['pyqentangle/cythonmodule/interpolate_nocheck.pyx'])
except ImportError:
    ext_modules = [Extension('pyqentangle.cythonmodule.interpolate_nocheck',
                             sources=['pyqentangle/cythonmodule/interpolate_nocheck.c'])]


def readme():
    with open('README.md') as f:
        return f.read()


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('# Quantum Entanglement in Python')
    return text[startpos:]


setup(name='pyqentangle',
      version="3.1.2",
      description="Quantum Entanglement in Python",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Chemistry",
          "License :: OSI Approved :: MIT License",
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
      install_requires=['numpy', 'tensornetwork'],
      tests_require=['unittest2', 'numpy', 'scipy',],
      ext_modules=ext_modules,
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
