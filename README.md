# Quantum Entanglement in Python

[![CircleCI](https://circleci.com/gh/stephenhky/pyqentangle.svg?style=svg)](https://circleci.com/gh/stephenhky/pyqentangle.svg)
[![GitHub release](https://img.shields.io/github/release/stephenhky/pyqentangle.svg?maxAge=3600)](https://github.com/stephenhky/pyqentangle/releases)
[![Documentation Status](https://readthedocs.org/projects/pyqentangle/badge/?version=latest)](https://pyqentangle.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/stephenhky/pyqentangle/shield.svg)](https://pyup.io/repos/github/stephenhky/pyqentangle/)
[![Python 3](https://pyup.io/repos/github/stephenhky/pyqentangle/python-3-shield.svg)](https://pyup.io/repos/github/stephenhky/pyqentangle/)
[![pypi](https://img.shields.io/pypi/v/pyqentangle.svg?maxAge=3600)](https://pypi.org/project/pyqentangle/)
[![download](https://img.shields.io/pypi/dm/pyqentangle.svg?maxAge=2592000&label=installs&color=%2327B1FF)](https://pypi.org/project/pyqentangle/)

## Version

The releases of `pyqentangle` 2.x.x are incompatible with previous releases.

The releases of `pyqentangle` 3.x.x are incompatible with previous releases.

The releases of `pyqentangle` 5.x.x are incompatible with previous releases.

Since release 3.1.0, the support for Python 2 was decomissioned.

## Installation

This package can be installed using `pip`.

```
>>> pip install pyqentangle
```

To use it, enter

```
>>> import pyqentangle
>>> import numpy as np
```

## Schmidt Decomposition for Discrete Bipartite States

We first express the bipartite state in terms of a tensor. For example, if the state is `|01>+|10>`, then express it as

```
>>> tensor = np.array([[0., np.sqrt(0.5)], [np.sqrt(0.5), 0.]])
```

To perform the Schmidt decompostion, just enter:

```
>>> pyqentangle.DiscreteSchmidtDecomposer(tensor).modes()
[DiscreteSchmidtMode(schmidt_coef=np.float64(0.7071067811865476), mode1=array([ 0., -1.]), mode2=array([-1., -0.])),
 DiscreteSchmidtMode(schmidt_coef=np.float64(0.7071067811865476), mode1=array([-1.,  0.]), mode2=array([-0., -1.]))]
 ```

In the returned list, for each element, there are the Schmidt coefficient, the component for first subsystem, and that
for the second subsystem.

## Schmidt Decomposition for Continuous Bipartite States

We can perform Schmidt decomposition on continuous systems too. For example, define the following normalized wavefunction:

```
>>> bipartite_wavefcn = pyqentangle.core.wavefunctions.AnalyticMultiDimWaveFunction(
        lambda x: np.exp(-0.5 * (x[0] + x[1]) ** 2) * np.exp(-(x[0] - x[1]) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)
    )
```

Then perform the Schmidt decomposition, 

```
>>> modes = pyqentangle.ContinuousSchmidtDecomposer(bipartite_wavefcn, -10., 10., -10., 10., keep=10).modes()
>>> modes
[ContinuousSchmidtMode(schmidt_coef=np.float64(0.9851714310094161), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd9a0f50>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd0d4f80>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(0.1690286950361957), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa8582d7dd0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd15e350>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(0.029000739207759557), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd0d2cf0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd0d1240>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(0.004975740210361184), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd4393d0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd4394f0>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(0.0008537020544076699), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439550>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439670>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(0.0001464721160848057), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd4396d0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd4397f0>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(2.5130642101174655e-05), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439850>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439970>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(4.311736522271967e-06), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd4399d0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439af0>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(7.397770324567384e-07), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439b50>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439c70>),
 ContinuousSchmidtMode(schmidt_coef=np.float64(1.2692567250818081e-07), wavefunction1=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439cd0>, wavefunction2=<pyqentangle.core.wavefunctions.InterpolatingWaveFunction object at 0x7fa6dd439df0>)]
```

where it describes the ranges of x1 and x2 respectively, and `keep=10` specifies only top 10 Schmidt modes are kept. 
Then we can read the Schmidt coefficients:

```
>>> from matplotlib import pyplot as plt
>>> [mode.schmidt_coef for mode in modes]
[np.float64(0.9851714310094161),
 np.float64(0.1690286950361957),
 np.float64(0.029000739207759557),
 np.float64(0.004975740210361184),
 np.float64(0.0008537020544076699),
 np.float64(0.0001464721160848057),
 np.float64(2.5130642101174655e-05),
 np.float64(4.311736522271967e-06),
 np.float64(7.397770324567384e-07),
 np.float64(1.2692567250818081e-07)]
```

The Schmidt functions can be plotted:
```
>>> xarray = np.linspace(-10., 10., 100)

    plt.subplot(3, 2, 1)
    plt.plot(xarray, modes[0].wavefunction1(xarray))
    plt.subplot(3, 2, 2)
    plt.plot(xarray, modes[0].wavefunction2(xarray))

    plt.subplot(3, 2, 3)
    plt.plot(xarray, modes[1].wavefunction1(xarray))
    plt.subplot(3, 2, 4)
    plt.plot(xarray, modes[1].wavefunction2(xarray))

    plt.subplot(3, 2, 5)
    plt.plot(xarray, modes[2].wavefunction1(xarray))
    plt.subplot(3, 2, 6)
    plt.plot(xarray, modes[2].wavefunction2(xarray))
```

![alt](https://github.com/stephenhky/pyqentangle/raw/master/fig/three_harmonic_modes.png)


## Useful Links

* Study of Entanglement in Quantum Computers: [https://datawarrior.wordpress.com/2017/09/20/a-first-glimpse-of-rigettis-quantum-computing-cloud/](https://datawarrior.wordpress.com/2017/09/20/a-first-glimpse-of-rigettis-quantum-computing-cloud/)
* Github page: [https://github.com/stephenhky/pyqentangle](https://github.com/stephenhky/pyqentangle)
* PyPI page: [https://pypi.python.org/pypi/pyqentangle/](https://pypi.python.org/pypi/pyqentangle/)
* Documentation: [http://pyqentangle.readthedocs.io/](http://pyqentangle.readthedocs.io/)
* RQEntangle: [https://CRAN.R-project.org/package=RQEntangle](https://CRAN.R-project.org/package=RQEntangle) (corresponding R library)

## Reference
* Artur Ekert, Peter L. Knight, "Entangled quantum systems and the Schmidt decomposition", *Am. J. Phys.* 63, 415 (1995).

## Acknowledgement
* [Hossein Seifoory](https://ca.linkedin.com/in/hosseinseifoory?trk=public_profile_card_url)
