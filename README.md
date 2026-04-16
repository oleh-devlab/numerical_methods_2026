# numerical_methods_2026

This repository contains implementations of classical numerical methods as part of a university course.
Almost every lab assignment focuses on applying mathematical methods to real or simulated data.

## Repository structure

- `lab0/` - introductory task
- `lab1/` - the use of cubic spline interpolation to construct a smooth elevation profile based on real-world geodata
- `lab2/` - using Newton's interpolation to approximate CPU load based on limited data
- `lab3/` - least-squares approximation for forecasting the average monthly temperature
- `lab4/` - numerical differentiation of a moisture model using finite differences with Runge-Romberg and Aitken refinements; step-size optimization and error analysis
- `lab5/` - numerical integration using Simpson’s method and its refinements by Runge-Römbberg, Eitken and an adaptive algorithm
- `lab6/` - solving linear systems using LU decomposition and Gaussian elimination

## Requirements

- Python 3.10+
    - numpy
    - matplotlib
    - requests

## Notes

- Input data files are stored inside each lab directory.
- Some labs contain helper scripts (for example, tabulation utilities) that can be run separately.