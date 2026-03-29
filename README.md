# numerical_methods_2026

This repository contains implementations of classical numerical methods as part of a university course.
Almost every lab assignment focuses on applying mathematical methods to real or simulated data.

## Repository structure

- `lab0/` - introductory task
- `lab1/` - the use of cubic spline interpolation to construct a smooth elevation profile based on real-world geodata
- `lab2/` - using Newton's interpolation to approximate CPU load based on limited data
- `lab3/` - least-squares approximation for forecasting the average monthly temperature

## Requirements

- Python 3.10+
    - numpy
    - matplotlib
    - requests

## Quick start

Run a lab:

```
python3 .\lab0\main.py
python3 .\lab1\main.py
python3 .\lab2\main.py
python3 .\lab3\main.py
```

## Notes

- Input data files are stored inside each lab directory.
- Some labs contain helper scripts (for example, tabulation utilities) that can be run separately.