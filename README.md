Spectral Characteristics of Kalman Filter Systems
=================================================

This bundle contains a module (DM93) and a collection of python scripts
illustrating important characteristics of the Kalman Filter using a
simple spectral advection model.

DM93 Module
-----------

The module is inspired by Daley, R and Ménard, R. (1993) which can be
found in the [American Meteorological Society](http://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1993)121%3C1554%3ASCOKFS%3E2.0.CO%3B2) and is intended to provide a simple heuristic data assimilation lab that one can modify and experiment with.

It contains 4 components:

-   `gridCls.py` describe the periodic grid class
-   `covarianceCls.py` describe the correlation and covariance classes
-   `spectralModelCls.py` describe the model
-   `DM93Lib.py` contains functions introduced in the aforementioned
    article

### Licence

The module is licenced under the GNU General Public License, please read
`LICENCE.txt`. Basically you can do whatever you want with it, modify,
redistribute, etc. As long as you propagate the licence and cite the
authors - Open Source rocks!

### Authors

-   Martin Deshaies-Jacques ([martin.deshaies-jacques@canada.ca](mailto:martin.deshaies-jacques@canada.ca))
-   Richard Ménard

### Dependencies

-   Python 2
-   Numpy
-   Matplotlib

These packages are readilly available on all major Linux distributions.

### Installation

The bundle does not contain a `setup.py` script; simply copy the
`./DM93/` folder in your work directory or add it to the `PYTHONPATH`.

Experiments and illustration of the theory
------------------------------------------
It is strongly suggested to read the article in order to understand the context and hypotheses underlying this module.
Main simplifying assumptions are:

- one dimensional linear equation
- regular and periodic analysis and forecast grid
- grid-points collocated observations (trivial observation operator)

All the scripts at the root level are self contained experiments or illustration of the theory exposed in the article to the exception that for most of them `config.py` is sourced to define constants and the periodic domain.

### **config.py**

-   defines space and time units
-   defines the assimilation window for model integration
-   defines the periodic domaine and spectral truncature
-   define the two physical parameters: the zonal wind speed and the dissipation coefficient

`config.py` is sourced in most of the other scripts (such that they share the parametrisation), but one can replace the statetment `execfile('config.py')` with explicit local definitions of the grid and parameters.

The grid defined in the script is an instance of `Grid` class defined in `./DM93/gridCls.py`.
This object also provide the discrete Fourier transform and its inverse.

All other scripts fall into two categories: illustration of the analytical developments in the aforementioned article or numerical experiments.


### Numerical experiments with the Kalman Filter

These are experiments involving actual computations using discrete Fourier transform and model integration dealing with correlation modeling, forecasting, analysis and assimilation.


#### **correlationModels.py**

Compare three isotropic and homogeneous correlation models:

-   Foar (first order auto-regressive model)
-   Soar (first order auto-regressive model)
-   Gaussian

It plots the correlations in function of distance and their normalized power spectra.
It also generates and plots one random spatial realisation of each of these models.

Worth mentioning, the correlation length is defined by the distance where the correlation reaches 1/sqrt(e).

Correlations models are instances of the `CorrModel` (itself deriving from `Covariance` class) and are defined in `./DM93/covarianceCls.py`.
All `Covariance` instances provide their matrix representation and a random generator.
`CorrModel` instances provide as well their radial correlation function, and their theoretical normalized power spectrum (as obtained on an infinite domain).


#### **sampleCorrelations.py**

Estimates correlation matrix with finite ensembles of perturbations and illustrates sampling noise inducing unphysical teleconnections.

Since the number of members is tightly constrained by integration cost in real atmospheric models, localisation is often used to circumvent this problem by restricting the sampled covariance on a compact support.

#### **propagation.py**

Integrate the numerical model and produce a trajectory using an `AdvectionDiffusionModel` instance.

`AdvectionDiffusionModel` is defined with a `Grid` instance, the physical parameters `U` and `nu` (in m/s) and an assimilation window `dt` in seconds.

One can change the grid or parameters setting either by modifying `config.py` or replacing the `execfile('config.py')` statement with explicit equivalent definitions.
Initial condition can also be changed, for instance, one could consider the evolution of a perturbation by initialising it with:

```python
from DM93 import Gaussian
B = Gaussian(grid, 300.)
ic = B.random()
```

But since the advection and diffusion model is non-dispersive, the initial condition is advected without deformation.
(A more difficult and lengthy exercice - requiring some python knowledge - would be to derive the `SpectralModel` class defined in `./DM93/spectralModelCls.py` and build a dispersive spectral model, then use it to propagate the perturbation and consider the evolution of its spectrum.)


#### **analysis.py**

Compute the analysis using optimal interpolation (direct inversion of B+R innovation matrix) and output the error reduction.

For both observation and forecast errors, statistics need to be provided:

-   correlation model
-   correlation length
-   bias
-   variance (constant on the domain)

By default (and as it is a common hypothesis in most context), the observation error are uncorrelated.
What would be the impact of having correlated observation errors? The impact of biases?


#### **kalmanFilter.py**

Run an assimilation cycle using the Kalman Filter; for each assimilation window: 

1.  observations are obtained (simulated from the truth plus a random realisation of R);
2.  an analysis is computed using optimal interpolation;
3.  the analysis if integrated using the model to produce the forecast;
4.  covariance matrices are propagated using the Kalman Filter equations;
5.  the forecast and covariance is then used for the next assimilation window.

Observation, forecast and model errors statistics need to be provided:

-   correlation model
-   correlation length
-   bias
-   variance (constant on the domain)

The script plots the truth and forecast trajectories as well as the forecast and analysis variances evolution in time.


#### **filterDivergence.py**

Illustrates the forecast variance evolution and the Kalman Filter divergence issue by comparing three assimilation experiments:

1.  A perfect model assimilation initialised with an imperfect initial condition
2.  A perfect model integration initialised with an imperfect initial condition (no assimilation)
3.  An imperfect model assimilation initialised with an imperfect initial condition

As in `kalmanFilter.py`, statistics need to be provided.

By default, only the variance comparison plot is produced, change `doPlotXPs = True` for all three experiments to produce trajectory plots.



----------------------------------------------


### Analytical spectral properties of the Kalman Filter

Theses scripts illustrates analytical relationships derived in the article or closely related to it.

#### **spectralVariance.py** 

Illustrates the Kalman Filter impact on variance at different scales for given forecast and observation error statistics.

Correlation models and length scales can be changed.


#### **stationarySolutions.py**

Illustrates the forecast variance convergence on the manifold for a given wavenumber.
The two stationary solutions are plotted, of which only one is stable and physical.

Iterates are identified by blue dots and numbers on the manifold.
As one can see by modifying `k` (the wavenumber), convergence slows with wavenumber.

(Reproduce the figure 1 from the article, section 2-a)



#### **assymptoticSolution.py**

Illustrates the assymptotical properties of the Kalman Filter.
Assymptotic forecast and analysis variance spectra are shown.
Also shown is the convergence rate in function of wavenumber.

How would correlated observation errors impact these properties?

(Reproduce the figure 2b from the article, section 3-a)


#### **viscosity.py**

Illustrates the impact of viscosity on the assymptotical variances and convergence rate spectra.

(Reproduce figure 3 from the articla, section 3-b)


#### **LcFromSpectra.py**

Correlation length is not defined univocally.
One definition is based on the curvature of a gaussian at the origin.
Based on this definition, it is possible to determine the correlation length from the power spectrum of the correlation.

This script use this definition to estimate the correlation length from three correlation models and compare them with the true one.

What explains that one model is better than the other?
What happens when the spectral resolution is better?



