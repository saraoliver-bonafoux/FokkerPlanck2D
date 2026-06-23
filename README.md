# Numerical integration of the 2D Fokker-Planck equation

This repository provides a Python implementation for the numerical solution of the two-dimensional Fokker–Planck equation:

$\partial_t P(x,y;t) = -\partial_x \left[ P(x,y;t) F_x(x,y) \right] -\partial_y \left[ P(x,y;t) F_y(x,y) \right] + \frac{D}{2} \partial_x^2 \left[ P(x,y;t) B_x(x,y) \right] + \frac{D}{2} \partial_y^2 \left[ P(x,y;t) B_y(x,y) \right].$

To numerically integrate this equation, we employ a finite-difference discretization in space combined with an alternating-direction implicit (ADI) scheme for time integration, which reduces the multidimensional problem to a sequence of one-dimensional tridiagonal systems.

Further details on the numerical scheme can be found in `NumericalScheme.pdf`.

The code is designed for reproducibility and can be adapted to a wide class of reaction–diffusion problems.

