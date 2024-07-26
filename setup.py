
from setuptools import setup
import numpy as np


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
      version="4.0.2",
      description="Quantum Entanglement in Python",
      long_description=package_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Chemistry",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Intended Audience :: Education"
      ],
      keywords="quantum physics Schmidt decompostion entanglement",
      url="https://github.com/stephenhky/pyqentangle",
      author="Kwan Yuet Stephen Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['pyqentangle', 'pyqentangle.quantumstates'],
      package_dir={'pyqentangle': 'pyqentangle'},
      include_dirs=[np.get_include()],
      setup_requires=['numpy', ],
      install_requires=install_requirements(),
      test_suite="test",
      include_package_data=True,
      zip_safe=False)
