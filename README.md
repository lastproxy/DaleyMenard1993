Spectral Characteristics of Kalman Filter Systems
=================================================

This bundle contains a module (DM93) and a collection of python scripts
illustrating important characteristics of the Kalman Filter using a
simple spectral advection model.

Its pupose is heuristic and scripts are meant to be fiddled with, change
this, change that and see what's going on! Always wondered what would
happen if observation errors where correlated?

DM93 Module
-----------

The module is inspired by Daley, R and Ménard, R. (1993) which can be
found in the American Meteorological Society
(\`[http://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1993)121%3C1554%3ASCOKFS%3E2.0.CO%3B2](http://journals.ametsoc.org/doi/abs/10.1175/1520-0493(1993)121%3C1554%3ASCOKFS%3E2.0.CO%3B2)\`\_).

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

-   Martin Deshaies-Jacques
    (\`[martin.deshaies-jacques@canada.ca](mailto:martin.deshaies-jacques@canada.ca)\`\_)
-   Richard Ménard

### Dependencies

-   Python 2
-   Numpy
-   Matplotlib

This packages are readilly available on all major Linux distributions.

### Installation

The bundle does not contain a `setup.py` script; simply copy the
`./DM93/` folder in your work directory or add it to the `PYTHONPATH`.

Experiments and illustration of the theory
------------------------------------------

... to be completed tomorrow!
