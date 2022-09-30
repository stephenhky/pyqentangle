
from setuptools import setup
import numpy as np
from Cython.Build import cythonize


ext_modules = cythonize(['pyqentangle/cythonmodule/interpolate_nocheck.pyx'])


def readme():
    with open('README.md') as f:
        return f.read()


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('# Quantum Entanglement in Python')
    return text[startpos:]


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


setup(name='pyqentangle',
      version="3.2.3",
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
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
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
      packages=['pyqentangle', 'pyqentangle.quantumstates', 'pyqentangle.cythonmodule'],
      package_dir={'pyqentangle': 'pyqentangle'},
      package_data={
          'pyqentangle': ['cythonmodule/*.c', 'cythonmodule/*.pyx']
      },
      include_dirs=[np.get_include()],
      setup_requires=['Cython', 'numpy', ],
      install_requires=install_requirements(),
      ext_modules=ext_modules,
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
