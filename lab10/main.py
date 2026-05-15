import numpy as np
import matplotlib.pyplot as plt

class LabODESolver:
    def __init__(self, f, exact_sol, a, b, y0):
        self.f = f
        self.exact_sol = exact_sol
        self.a = a
        self.b = b
        self.y0 = y0

    def rk4_step(self, x, y, h):
        k1 = self.f(x, y)
        k2 = self.f(x + h/2, y + h * k1 / 2)
        k3 = self.f(x + h/2, y + h * k2 / 2)
        k4 = self.f(x + h, y + h * k3)
        return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def rk4_fixed(self, h):
        x_vals, y_vals = [], []
        x, y = self.a, self.y0
        
        while x <= self.b + 1e-9:
            x_vals.append(x)
            y_vals.append(y)
            y = self.rk4_step(x, y, h)
            x += h
            
        return np.array(x_vals), np.array(y_vals)

    def rk4_auto(self, h0, epsilon):
        x_vals, y_vals, h_vals = [], [], []
        x, y, h = self.a, self.y0, h0
        k_const = 32 # 2^(s+1) для 4-го порядку
        
        while x < self.b:
            if x + h > self.b: h = self.b - x
            
            y_h = self.rk4_step(x, y, h)
            y_half_1 = self.rk4_step(x, y, h/2)
            y_half_2 = self.rk4_step(x + h/2, y_half_1, h/2)
            
            error = (16 / 15) * abs(y_half_2 - y_h)
            
            if error > epsilon:
                h /= 2 
            else:
                x_vals.append(x)
                y_vals.append(y)
                h_vals.append(h)
                
                x += h
                y = y_half_2 
                
                if error <= epsilon / k_const:
                    h *= 2
                    
        x_vals.append(x); y_vals.append(y); h_vals.append(h)
        return np.array(x_vals), np.array(y_vals), np.array(h_vals)

    def adams2_fixed(self, h, eps_iter=1e-5):
        x_vals = np.arange(self.a, self.b + h, h)
        n = len(x_vals)
        y_vals = np.zeros(n)
        y_pred_vals = np.zeros(n)
        y_corr_vals = np.zeros(n)
        
        y_vals[0] = self.y0
        y_pred_vals[0] = self.y0
        y_corr_vals[0] = self.y0
        
        if n > 1:
            y_vals[1] = self.rk4_step(x_vals[0], y_vals[0], h)
            y_pred_vals[1] = y_vals[1]
            y_corr_vals[1] = y_vals[1]
            
        for i in range(1, n - 1):
            x_n, x_prev, x_next = x_vals[i], x_vals[i-1], x_vals[i+1]
            f_n, f_prev = self.f(x_n, y_vals[i]), self.f(x_prev, y_vals[i-1])
            
            y_pred = y_vals[i] + (h / 2) * (3 * f_n - f_prev)
            y_pred_vals[i+1] = y_pred
            
            y_mod = y_pred + (5 / 6) * (y_corr_vals[i] - y_pred_vals[i])
            
            y_corr_prev = y_mod
            for _ in range(2): 
                y_corr = y_vals[i] + (h / 2) * (self.f(x_next, y_corr_prev) + f_n)
                if abs(y_corr - y_corr_prev) <= eps_iter:
                    break
                y_corr_prev = y_corr
                
            y_corr_vals[i+1] = y_corr
            
            y_vals[i+1] = y_corr - (1 / 6) * (y_corr - y_pred)
            
        return x_vals, y_vals, y_pred_vals, y_corr_vals

    def adams2_auto(self, h0, epsilon):
        x_vals, y_vals, h_vals = [], [], []
        x, h = self.a, h0
        
        y_prev, x_prev = self.y0, x
        x += h
        y_curr = self.rk4_step(x_prev, y_prev, h)
        
        x_vals.extend([x_prev, x])
        y_vals.extend([y_prev, y_curr])
        h_vals.extend([h, h])
        
        y_pred_prev = y_curr
        y_corr_prev_step = y_curr
        
        while x < self.b:
            if x + h > self.b: h = self.b - x
                
            f_n, f_prev = self.f(x, y_curr), self.f(x_prev, y_prev)
            
            y_pred = y_curr + (h / 2) * (3 * f_n - f_prev)
            
            y_mod = y_pred + (5 / 6) * (y_corr_prev_step - y_pred_prev)
            
            y_corr_iter = y_mod
            for _ in range(2):
                y_corr = y_curr + (h / 2) * (self.f(x + h, y_corr_iter) + f_n)
                if abs(y_corr - y_corr_iter) <= 1e-6:
                    break
                y_corr_iter = y_corr
            
            error_est = abs(y_corr - y_pred) / 6.0
            
            if error_est > epsilon:
                h /= 2
                y_prev, x_prev = y_curr, x
                y_curr = self.rk4_step(x_prev, y_prev, h)
                x += h
                x_vals.append(x); y_vals.append(y_curr); h_vals.append(h)
            
                y_pred_prev, y_corr_prev_step = y_curr, y_curr
                continue 
            else:
                y_pred_prev = y_pred
                y_corr_prev_step = y_corr
                
                x_prev, y_prev = x, y_curr
                x += h
                
                y_curr = y_corr - (1 / 6) * (y_corr - y_pred)
                
                x_vals.append(x); y_vals.append(y_curr); h_vals.append(h)
                
                if error_est <= epsilon / 8:
                    h *= 2
                    y_prev, x_prev = y_curr, x
                    y_curr = self.rk4_step(x_prev, y_prev, h)
                    x += h
                    x_vals.append(x); y_vals.append(y_curr); h_vals.append(h)
                    y_pred_prev, y_corr_prev_step = y_curr, y_curr
                    
        return np.array(x_vals), np.array(y_vals), np.array(h_vals)

if __name__ == "__main__":
    h_fixed = 0.01
    eps_rk4 = 1e-4
    eps_adams = 1e-3

    solver = LabODESolver(
        f=lambda x, y: -2 * x * y, # Права частина рівняння y' = f(x,y)
        exact_sol=lambda x: np.exp(-x**2), # Точний розв'язок
        a=0.0, b=3.0, y0=1.0
    )

    x_rk_fix, y_rk_fix = solver.rk4_fixed(h_fixed)
    y_exact_rk = solver.exact_sol(x_rk_fix)
    err_exact_rk = np.abs(y_rk_fix - y_exact_rk)
    
    err_runge_rk = np.zeros_like(x_rk_fix)
    for i in range(len(x_rk_fix)-1):
        yh = solver.rk4_step(x_rk_fix[i], y_rk_fix[i], h_fixed)
        yh_half = solver.rk4_step(x_rk_fix[i] + h_fixed/2, 
                                  solver.rk4_step(x_rk_fix[i], y_rk_fix[i], h_fixed/2), 
                                  h_fixed/2)
        err_runge_rk[i+1] = (16/15) * abs(yh_half - yh)
        
    x_rk_auto, y_rk_auto, h_rk_auto = solver.rk4_auto(h0=h_fixed, epsilon=eps_rk4)

    x_ad_fix, y_ad_fix, y_ad_pred, y_ad_corr = solver.adams2_fixed(h_fixed)
    y_exact_ad = solver.exact_sol(x_ad_fix)
    err_exact_ad = np.abs(y_ad_fix - y_exact_ad)
    
    err_est_ad = np.abs(y_ad_corr - y_ad_pred) / 6.0
    
    x_ad_auto, y_ad_auto, h_ad_auto = solver.adams2_auto(h0=h_fixed, epsilon=eps_adams)

    # --- Побудова графіків ---
    # Метод Рунге-Кутта
    plt.figure("Частина 2: Метод Рунге-Кутта", figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_rk_fix, solver.exact_sol(x_rk_fix), label="Точний", lw=3, alpha=0.5)
    plt.plot(x_rk_auto, y_rk_auto, '--', label=f"Авто РК4 (eps={eps_rk4})")
    plt.title("Розв'язок")
    plt.grid(); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x_rk_fix, err_exact_rk, '-', label="Точна похибка")
    plt.plot(x_rk_fix, err_runge_rk, '--', label="За правилом Рунге")
    plt.yscale('log')
    plt.title(f"Локальні похибки (h={h_fixed})")
    plt.grid(); plt.legend()

    plt.subplot(1, 3, 3)
    plt.step(x_rk_auto, h_rk_auto, where='pre', color='green')
    plt.title("Зміна кроку h(x)")
    plt.grid()

    plt.tight_layout()

    # Метод Адамса
    plt.figure("Частина 1: Метод Адамса", figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_ad_fix, solver.exact_sol(x_ad_fix), label="Точний", lw=3, alpha=0.5)
    plt.plot(x_ad_auto, y_ad_auto, '--', label=f"Авто Адамс (eps={eps_adams})")
    plt.title("Розв'язок")
    plt.grid(); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x_ad_fix[2:], err_exact_ad[2:], '-', label="Точна похибка")
    plt.plot(x_ad_fix[2:], err_est_ad[2:], '--', label="Оцінка 1/6|Y_kop - Y_pr|")
    plt.yscale('log')
    plt.title(f"Локальні похибки (h={h_fixed})")
    plt.grid(); plt.legend()

    plt.subplot(1, 3, 3)
    plt.step(x_ad_auto, h_ad_auto, where='pre', color='purple')
    plt.title("Зміна кроку h(x)")
    plt.grid()

    plt.tight_layout()
    plt.show()