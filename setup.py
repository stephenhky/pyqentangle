from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyqentangle',
      version="0.16",
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
      install_requires=[
          'numpy',
      ],
      include_package_data=True,
      zip_safe=False)
