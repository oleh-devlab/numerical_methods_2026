import numpy as np
import matplotlib.pyplot as plt
import math

# Задана функція
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

C1 = 240 / np.pi
C2 = 2.5 * np.sqrt(5 * np.pi)
SQRT_0_2 = np.sqrt(0.2)

# Аналітична первісна для функції f(x)
def F(x):
    return 50 * x - C1 * np.cos(np.pi * x / 12) + C2 * math.erf(SQRT_0_2 * (x - 12))

# --- П. 3: Метод Сімпсона ---
def simpson_integral(func, a, b, N):
    if N % 2 != 0:
        print("Для методу Сімпсона число розбиттів N має бути парним.")
        return None
        
    h = (b - a) / N
    x_vals = np.linspace(a, b, N + 1)
    y_vals = func(x_vals)
    
    sum_odd = np.sum(y_vals[1:-1:2])
    sum_even = np.sum(y_vals[2:-2:2])
    
    return (h / 3) * (y_vals[0] + 4 * sum_odd + 2 * sum_even + y_vals[-1])

def aitken_refinement(I_N0, I_N0_half, I_N0_quarter):
     denom = 2 * I_N0_half - (I_N0 + I_N0_quarter)
     if denom != 0:
         return (I_N0_half**2 - I_N0 * I_N0_quarter) / denom
     else:
         return I_N0_half

# Оцінка порядку методу p
def aitken_order(I_N0, I_N0_half, I_N0_quarter):
     denom_p = I_N0_half - I_N0
     num_p = I_N0_quarter - I_N0_half
     if denom_p != 0 and (num_p / denom_p) > 0:
         return math.log(abs(num_p / denom_p)) / math.log(2)
     else:
         return float('nan')

def adaptive_simpson(func, a, b, tol):
    calls = [3]
    m = (a + b) / 2.0
    fa, fm, fb = func(a), func(m), func(b)

    max_depth = 50
    
    # Стек зберігає кортежі з параметрами для кожного інтервалу:
    # (a, b, tol, fa, fm, fb, поточна_глибина)
    stack = [(a, b, tol, fa, fm, fb, 0)]
    total_integral = 0.0
    
    while stack:
        # поведінка LIFO імітує рекурсію
        cur_a, cur_b, cur_tol, cur_fa, cur_fm, cur_fb, depth = stack.pop()
        
        h = cur_b - cur_a
        m = (cur_a + cur_b) / 2.0
        
        m1 = (cur_a + m) / 2.0
        m2 = (m + cur_b) / 2.0
        
        fm1 = func(m1)
        fm2 = func(m2)
        calls[0] += 2
        
        I1 = (h / 6.0) * (cur_fa + 4.0 * cur_fm + cur_fb)
        
        I_left = (h / 12.0) * (cur_fa + 4.0 * fm1 + cur_fm)
        I_right = (h / 12.0) * (cur_fm + 4.0 * fm2 + cur_fb)
        I2 = I_left + I_right
        
        # Умова зупинки: досягнуто точність або досягнуто ліміт поділу інтервалу
        if abs(I1 - I2) <= 15 * cur_tol or depth >= max_depth:
            total_integral += I2 + (I2 - I1) / 15.0
        else:
            stack.append((m, cur_b, cur_tol / 2.0, cur_fm, fm2, cur_fb, depth + 1))
            stack.append((cur_a, m, cur_tol / 2.0, cur_fa, fm1, cur_fm, depth + 1))
            
    result = total_integral
    return result, calls[0]

# Інтервал інтегрування
a = 0
b = 24

# --- П. 2: Точне значення інтегралу ---
I0 = F(b) - F(a)
print(f"--- Пункт 2 ---")
print(f"Точне значення інтегралу I0: {I0:.15f}\n")

# --- П. 4: Дослідження залежності похибки від N ---
target_eps = 1e-12
N_values = []
errors = []

N_opt = None
eps_opt = None

for N in range(10, 1002, 2):
    I_N = simpson_integral(f, a, b, N)
    current_error = abs(I_N - I0)
    
    N_values.append(N)
    errors.append(current_error)
    
    if current_error <= target_eps and N_opt is None:
        N_opt = N
        eps_opt = current_error

print(f"--- Пункт 4 ---")
if N_opt is not None:
    print(f"Оптимальне число розбиттів (N_opt): {N_opt}")
    print(f"Отримана точність (eps_opt): {eps_opt:.15e}")
else:
    print(f"Задана точність {target_eps} не досягнута при N <= 1000")
    N_opt = 1000

# --- П. 5: Обчислення похибки при N0 ---
N0_raw = N_opt / 10
N0 = max(8, int(round(N0_raw / 8.0)) * 8)

I_N0 = simpson_integral(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print(f"\n--- Пункт 5 ---")
print(f"Розрахункове N_opt / 10 = {N0_raw:.2f}. Вибрано N0 (кратне 8) = {N0}")
print("Метод Сімпсона:")
print(f"Значення інтегралу I(N0): {I_N0:.15f}")
print(f"Похибка обчислення eps0: {eps0:.15e}\n")

# --- П. 6: Метод Рунге-Ромберга ---
N0_half = N0 // 2
I_N0_half = simpson_integral(f, a, b, N0_half)

I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I0)

print(f"--- Пункт 6: Метод Рунге-Ромберга ---")
print(f"Значення інтегралу I(N0/2): {I_N0_half:.15f}")
print(f"Уточнене значення інтегралу I_R: {I_R:.15f}")
print(f"Похибка Рунге-Ромберга epsR: {epsR:.15e}")

if epsR > 0:
    improvement = eps0 / epsR
    print(f"Точність покращилась у {improvement:.2f} разів у порівнянні з методом Сімпсона\n")

# --- П. 7: Метод Ейткена ---
N0_quarter = N0 // 4
I_N0_quarter = simpson_integral(f, a, b, N0_quarter)
     
I_E = aitken_refinement(I_N0, I_N0_half, I_N0_quarter)
p = aitken_order(I_N0, I_N0_half, I_N0_quarter)

epsE = abs(I_E - I0)

print(f"--- Пункт 7: Метод Ейткена ---")
print(f"Порядок методу p (оцінка): {p:.4f}")
print(f"Похибка за Ейткеном epsE: {epsE:.15e}\n")

if epsE > 0:
    improvement = eps0 / epsE
    print(f"Точність покращилась у {improvement:.2f} разів у порівнянні з методом Сімпсона\n")

# --- П. 9: Адаптивний алгоритм ---
print(f"--- Пункт 9: Адаптивний алгоритм ---")
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
adaptive_errors = []
eval_counts = []

print(f"{'Задана точність (delta)':^25} | {'Абс. похибка':^20} | {'К-сть обчислень f(x)':<20} | Покращення від eps0")
print("-" * 93)

for tol in tolerances:
    res, count = adaptive_simpson(f, a, b, tol)
    err = abs(res - I0)
    adaptive_errors.append(err)
    eval_counts.append(count)
    print(f"{tol:^25.0e} | {err:^20.2e} | {count:<20} | {eps0/err:.1e}")

# --- Побудова графіків  ---
# (Пункт 1)
x = np.linspace(a, b, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Графік залежності похибки від N (Пункт 4)
plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, label=r'$\epsilon(N) = |I(N) - I_0|$', color='blue')

plt.axhline(y=target_eps, color='red', linestyle='--', label=fr'Задана точність $\epsilon={target_eps}$')

if N_opt is not None and eps_opt is not None:
    plt.scatter([N_opt], [eps_opt], color='red', zorder=5)

plt.yscale('log')
plt.title('Залежність похибки методу Сімпсона від числа розбиттів N')
plt.xlabel('Число розбиттів (N)')
plt.ylabel('Абсолютна похибка (логарифмічна шкала)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()

# Адаптивний алгоритм (Пункт 9)
plt.figure(figsize=(12, 8))
plt.plot(tolerances, eval_counts, marker='s', color='darkcyan')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.title('Адаптивний метод Сімпсона: кількість обчислень f(x) залежно від заданої точності')
plt.xlabel('Заданий параметр (delta)')
plt.ylabel('Кількість обчислень f(x)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()