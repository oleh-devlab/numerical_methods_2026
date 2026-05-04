import csv
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def tabulate_function(func, a, b, h, filename="tabulated_data.txt"):
    x_vals = np.arange(a, b + h, h)
    y_vals = func(x_vals)
    
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["x", "F(x)"])
        for x, y in zip(x_vals, y_vals):
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])
            
    return x_vals, y_vals

def read_tabulated_data(filename="tabulated_data.txt"):
    """Зчитує табульовані дані з файлу."""
    x, y = [], []
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Пропуск заголовка
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)

def find_approximate_roots(x_vals, y_vals):
    roots_approx = []
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i+1] <= 0:
            trend = "Зростає" if y_vals[i+1] > y_vals[i] else "Спадає"
            roots_approx.append({
                'x0': (x_vals[i] + x_vals[i+1]) / 2,
                'trend': trend
            })
    return roots_approx

def write_polynomial_coeffs(coeffs, filename="poly_coeffs.txt"):
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(",".join(map(str, coeffs)))

def read_polynomial_coeffs(filename="poly_coeffs.txt"):
    filepath = os.path.join(BASE_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        coeffs = [float(c) for c in line.split(',')]
    return coeffs