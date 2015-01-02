sirs_gsp
========

sirs_gsp = "SIRS Gillespie"

Python scripts for performing continuous-time stochastic simulations of
SIRS-type disease dynamics in fully mixed populations.

This module allows us to implement continuous-time stochastic simulations 
of SIR-type disease dynamics.  We allow for SI, SIS, SIR, and SIRS models
in fully mixed systems.  The simulations use Gillespie's exact algorithm.

Thanks to Sarabjeet Singh, who provided the inspiration for this code 
with Gillespie_messy.py.

This repository contains:

sirs_gsp.py - The python package, containing scripts for performing
the SIRS simulations

sirs_gsp_demo.ipynb - iPython notebook demonstrating the use of this
package

sirs_gsp.jl - Julia version of sirs_gsp.py, which includes a multi-
threaded version of the code for best performance.

README.md - what you're reading right now
