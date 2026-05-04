import numpy as np
import matplotlib.pyplot as plt
from utils import tabulate_function, read_tabulated_data, find_approximate_roots, write_polynomial_coeffs, read_polynomial_coeffs

def plot_polynomial():
    x = np.linspace(-2, 3, 400)
    y = x**3 - x**2 - x - 2

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='$y = x^3 - x^2 - x - 2$', color='blue')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    plt.plot(2, 0, 'ro', label='Дійсний корінь ($x=2$)')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Графік алгебраїчного рівняння (пошук дійсних коренів)')
    plt.legend()
    
    plt.show()

class NonlinearSolver:
    def __init__(self, F, dF, ddF, eps=1e-10):
        self.F = F
        self.dF = dF
        self.ddF = ddF
        self.eps = eps

    def check_stop_criteria(self, x_new, x_old):
        return abs(self.F(x_new)) < self.eps and abs(x_new - x_old) < self.eps

    def simple_iteration(self, x0, tau):
        x_old = x0
        iterations = 0
        while True:
            x_new = x_old + tau * self.F(x_old)
            iterations += 1
            if self.check_stop_criteria(x_new, x_old) or iterations > 1000:
                break
            x_old = x_new
        return x_new, iterations

    def newton(self, x0):
        x_old = x0
        iterations = 0
        while True:
            x_new = x_old - self.F(x_old) / self.dF(x_old)
            iterations += 1
            if self.check_stop_criteria(x_new, x_old) or iterations > 1000:
                break
            x_old = x_new
        return x_new, iterations

    def chebyshev(self, x0):
        x_old = x0
        iterations = 0
        while True:
            F_val = self.F(x_old)
            dF_val = self.dF(x_old)
            ddF_val = self.ddF(x_old)
            term1 = F_val / dF_val
            term2 = (0.5 * (F_val**2) * ddF_val) / (dF_val**3)
            
            x_new = x_old - term1 - term2
            iterations += 1
            if self.check_stop_criteria(x_new, x_old) or iterations > 1000:
                break
            x_old = x_new
        return x_new, iterations

    def secant(self, x0, x1):
        iterations = 0
        while True:
            F0 = self.F(x0)
            F1 = self.F(x1)
            if abs(F1 - F0) < 1e-15:
                break
            
            x_new = x1 - F1 * (x1 - x0) / (F1 - F0)
            iterations += 1
            if self.check_stop_criteria(x_new, x1) or iterations > 1000:
                break
            x0, x1 = x1, x_new
        return x_new, iterations

    def parabola(self, x0, x1, x2):
        import cmath
        iterations = 0
        while True:
            f0, f1, f2 = self.F(x0), self.F(x1), self.F(x2)
            
            f10 = (f1 - f0) / (x1 - x0)
            f21 = (f2 - f1) / (x2 - x1)
            f210 = (f21 - f10) / (x2 - x0)
            
            a = f210
            b = (x2 - x1) * f210 + f21
            c = f2
            
            D = cmath.sqrt(b**2 - 4 * a * c)
            
            if a == 0:
                delta = -c / b
            else:
                d1 = (-b + D) / (2 * a)
                d2 = (-b - D) / (2 * a)
                delta = d1 if abs(d1) < abs(d2) else d2
            
            x_new = (x2 + delta).real
            iterations += 1
            
            if self.check_stop_criteria(x_new, x2) or iterations > 1000:
                break
                
            x0, x1, x2 = x1, x2, x_new
            
        return x_new, iterations
    
    def inverse_interpolation(self, x0, x1, x2):
        iterations = 0
        while True:
            y0, y1, y2 = self.F(x0), self.F(x1), self.F(x2)
            
            term0 = (y1 * y2) / ((y0 - y1) * (y0 - y2)) * x0
            term1 = (y0 * y2) / ((y1 - y0) * (y1 - y2)) * x1
            term2 = (y0 * y1) / ((y2 - y0) * (y2 - y1)) * x2
            
            x_new = term0 + term1 + term2
            iterations += 1
            
            if self.check_stop_criteria(x_new, x2) or iterations > 1000:
                break
                
            x0, x1, x2 = x1, x2, x_new
        return x_new, iterations


class AlgebraicSolver:
    def __init__(self, coeffs, eps=1e-10):
        self.coeffs = coeffs
        self.eps = eps
        self.m = len(coeffs) - 1

    def poly_value(self, x):
        val = 0
        for i, a in enumerate(self.coeffs):
            val += a * (x**i)
        return val

    def newton_horner(self, x0):
        x_n = x0
        iterations = 0
        
        while True:
            b = [0] * (self.m + 1)
            b[self.m] = self.coeffs[self.m]
            for i in range(self.m - 1, -1, -1):
                b[i] = self.coeffs[i] + x_n * b[i+1]
            
            c = [0] * (self.m + 1)
            c[self.m] = b[self.m]
            for i in range(self.m - 1, 0, -1):
                c[i] = b[i] + x_n * c[i+1]
                
            b0 = b[0]
            c1 = c[1]
            
            x_next = x_n - b0 / c1
            iterations += 1
            
            if abs(self.poly_value(x_next)) < self.eps and abs(x_next - x_n) < self.eps:
                break
                
            x_n = x_next
        return x_next, iterations

    def lin_method(self, alpha0, beta0):
        alpha, beta = alpha0, beta0
        iterations = 0
        
        while True:
            p0 = -2 * alpha
            q0 = alpha**2 + beta**2
            
            b = [0] * (self.m + 1)
            b[self.m] = self.coeffs[self.m]
            b[self.m - 1] = self.coeffs[self.m - 1] - p0 * b[self.m]
            
            for i in range(self.m - 2, 1, -1):
                b[i] = self.coeffs[i] - p0 * b[i+1] - q0 * b[i+2]
                
            b2, b3 = b[2], b[3] if self.m >= 3 else 0
            a0, a1 = self.coeffs[0], self.coeffs[1]
            
            q1 = a0 / b2
            p1 = (a1 * b2 - a0 * b3) / (b2**2)
            
            alpha_new = -p1 / 2
            
            if q1 - alpha_new**2 < 0:
                beta_new = 0
            else:
                beta_new = np.sqrt(q1 - alpha_new**2)
                
            iterations += 1
            
            if abs(alpha_new - alpha) <= self.eps and abs(beta_new - beta) <= self.eps:
                break
                
            alpha, beta = alpha_new, beta_new
            
            if iterations > 1000:
                break
                
        return alpha_new, beta_new, iterations

def main():
    print("--- Частина 1-4: Нелінійні рівняння ---")
    F = lambda x: np.sin(x) - 0.5 * x
    dF = lambda x: np.cos(x) - 0.5
    ddF = lambda x: -np.sin(x)

    # 1. Табуляція
    a, b_val, h = -1, 3, 0.1
    x_vals, y_vals = tabulate_function(F, a, b_val, h, "transcendental_data.txt")
    approx_roots = find_approximate_roots(x_vals, y_vals)
    
    print("Знайдені наближені корені:")
    for i, r in enumerate(approx_roots):
        print(f"Корінь {i+1}: x0 ~ {r['x0']:.4f} (Функція {r['trend'].lower()})")

    # 2-4. Розв'язок заданою точністю
    solver = NonlinearSolver(F, dF, ddF, eps=1e-10)
    
    roots_to_test = [approx_roots[0]['x0'], approx_roots[1]['x0']]
    
    for i, x0 in enumerate(roots_to_test):
        print(f"\n--- Дослідження кореня {i+1} (Початкове наближення x0 = {x0:.4f}) ---")
        
        tau = -0.5 if dF(x0) > 0 else 0.5
        
        x_simp, it_simp = solver.simple_iteration(x0, tau)
        print(f"Проста ітерація: x* = {x_simp:.10f}, Ітерацій: {it_simp}")
        
        x_newt, it_newt = solver.newton(x0)
        print(f"Метод Ньютона:   x* = {x_newt:.10f}, Ітерацій: {it_newt}")
        
        x_cheb, it_cheb = solver.chebyshev(x0)
        print(f"Метод Чебишева:  x* = {x_cheb:.10f}, Ітерацій: {it_cheb}")
        
        x_sec, it_sec = solver.secant(x0 - 0.1, x0 + 0.1)
        print(f"Метод хорд:      x* = {x_sec:.10f}, Ітерацій: {it_sec}")
        
        x_par, it_par = solver.parabola(x0 - 0.1, x0, x0 + 0.1)
        print(f"Метод парабол:       x* = {x_par:.10f}, Ітерацій: {it_par}")
        
        x_inv, it_inv = solver.inverse_interpolation(x0 - 0.1, x0, x0 + 0.1)
        print(f"Зворотня інтерп.:x* = {x_inv:.10f}, Ітерацій: {it_inv}")


    print("\n--- Частина 5-9: Алгебраїчні рівняння ---")
    # Рівняння: x^3 - x^2 - x - 2 = 0
    plot_polynomial()
    # Дійсний корінь: 2. Комплексні: -0.5 +/- i*sqrt(3)/2
    coeffs = [-2, -1, -1, 1]
    write_polynomial_coeffs(coeffs, "poly_coeffs.txt")
    
    read_coeffs = read_polynomial_coeffs("poly_coeffs.txt")
    alg_solver = AlgebraicSolver(read_coeffs, eps=1e-10)
    
    print(f"Коефіцієнти многочлена: {read_coeffs}")
    
    # 8. Метод Ньютона-Горнера
    x_real, it_real = alg_solver.newton_horner(x0=1.5)
    print(f"\nМетод Ньютона-Горнера (дійсний корінь):")
    print(f"x* = {x_real:.10f}, Ітерацій: {it_real}")
    
    # 9. Метод Ліна
    alpha0, beta0 = -0.4, 0.8 # Початкове наближення
    alpha_res, beta_res, it_comp = alg_solver.lin_method(alpha0, beta0)
    print(f"\nМетод Ліна (комплексні корені):")
    print(f"x* = {alpha_res:.10f} +/- i*{beta_res:.10f}, Ітерацій: {it_comp}")


if __name__ == "__main__":
    main()