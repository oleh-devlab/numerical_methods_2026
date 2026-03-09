import numpy as np
import matplotlib.pyplot as plt
import math

from tabulate_data import BASE_DIR, read_data, read_data_from_txt

# -------------------------------
# Класи, функції та отримання даних
# -------------------------------

class NewtonInterpolator:
	def __init__(self, x, y, verbose=False):
		self.x_data = x
		self.y = y
		self.verbose = verbose
		self.coef = self._divided_diff()

	# Функція для обчислення таблиці розділених різниць
	def _divided_diff(self):
		x = self.x_data
		y = self.y

		n = len(y)
		coef = np.copy(y).astype(float)
		
		if self.verbose:
			print("\nІтерації обчислення розділених різниць:")
			for j in range(1, n):
				coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
				print(f"Порядок {j}: {np.round(coef[j:n], 4)}")
		else:
			for j in range(1, n):
				coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
		return coef
	
	# Функція для обчислення многочлена Ньютона в точці
	def newton_polynomial(self, x):
		n = len(self.coef)
		p = self.coef[-1]
		for k in range(2, n+1):
			p = self.coef[-k] + (x - self.x_data[-k]) * p	
		return p
	
	def omega_function(self, x):
		result = 1.0
		for xi in self.x_data:
			result *= (x - xi)
		return result
	
	def get_graph_vals(self):
		x_vals = np.linspace(self.x_data.min(), self.x_data.max(), 100)
		y_vals = [self.newton_polynomial(xi) for xi in x_vals]

		return np.array(x_vals), np.array(y_vals)

class FactorialInterpolator:
    def __init__(self, x, y, verbose=False):
        self.x_data = x
        self.y_data = y
        self.n = len(y)
        self.verbose = verbose
        # Обчислюємо крок (припускаючи, що вузли рівновіддалені)
        self.h = x[1] - x[0]
        self.diffs = self._forward_differences()

    def _forward_differences(self):
        # Матриця для звичайних скінченних різниць
        diffs = np.zeros((self.n, self.n))
        diffs[:, 0] = self.y_data
        
        for j in range(1, self.n):
            for i in range(self.n - j):
                # Формула скінченної різниці: f(x_{i+1}) - f(x_i)
                diffs[i, j] = diffs[i+1, j-1] - diffs[i, j-1]
        
        if self.verbose:
            print("\nСкінченні різниці (верхній рядок для факторіального многочлена):")
            for j in range(1, self.n):
                print(f"Порядок {j}: {diffs[0, j]:.4f}")
                
        return diffs[0, :] # Для формули потрібен лише верхній рядок

    def falling_factorial(self, t, k):
        # Обчислення спадного факторіала t^(k) = t(t-1)...(t-k+1)
        if k == 0:
            return 1.0
        result = 1.0
        for i in range(k):
            result *= (t - i)
        return result

    def predict(self, x_val):
        # Перехід до нової змінної t
        t = (x_val - self.x_data[0]) / self.h
        p = 0.0
        
        # Формула: Сума ( Delta^k f(0) / k! ) * t^(k)
        for k in range(self.n):
            term = (self.diffs[k] / math.factorial(k)) * self.falling_factorial(t, k)
            p += term
        return p

x, y = read_data("data.csv")
x = np.array(x)
y = np.array(y)

def get_reduced_list(k, x_full, y_full):
    indices = np.linspace(0, len(x_full) - 1, k, dtype=int)
    x_k = x_full[indices]
    y_k = y_full[indices]
    
    print(f"=== {k} вузлів ===")
    print(f"Вибрані x: {x_k}")
    print(f"Вибрані y: {y_k}")
    return x_k, y_k



# -------------------------------
# Аналіз та графіки
# -------------------------------

# Створюємо інтерполяційні моделі для даних з варіанту 2

print(f"=== {len(x)} вузлів ===")
model_5 = NewtonInterpolator(x, y, verbose=True)

x_k, y_k = get_reduced_list(4, x, y)
model_4 = NewtonInterpolator(x_k, y_k, verbose=True)

target_rps = 600
cpu_4 = model_4.newton_polynomial(target_rps)
cpu_5 = model_5.newton_polynomial(target_rps)

# Похибка (різниця між 5 та 4 вузлами)
error = abs(cpu_5 - cpu_4)

# Обчислення омега для 5 вузлів у точці 600
omega_val = model_5.omega_function(target_rps)

print("\n--- Факторіальний многочлен (на рівновіддалених вузлах) ---")

# Факторіальний многочлен (на рівновіддалених вузлах)
x_eq = np.linspace(x.min(), x.max(), 5)
y_eq = np.array([model_5.newton_polynomial(xi) for xi in x_eq])

model_fact = FactorialInterpolator(x_eq, y_eq, verbose=True)
cpu_fact = model_fact.predict(target_rps)
print("-" * 30)

# Тестуємо у проміжку
x_vals_5, y_vals_5 = model_5.get_graph_vals()
x_vals_4, y_vals_4 = model_4.get_graph_vals()

# Точки для факторіального многочлена
y_vals_fact = np.array([model_fact.predict(xi) for xi in x_vals_5])

error_v2 = np.abs(y_vals_5 - y_vals_4)

x_eq = np.linspace(x.min(), x.max(), 5)
y_eq = np.array([model_5.newton_polynomial(xi) for xi in x_eq])

# Побудова графіка
plt.plot(x, y, 'ro', label="Вузли")
plt.plot(x_vals_5, y_vals_5, 'b-', label="Многочлен Ньютона (усі вузли)")
plt.plot(x_vals_4, y_vals_4, 'g--', label="4 вузли")

plt.plot(x_vals_5, y_vals_fact, 'c:', linewidth=3, label="Факторіальний многочлен")
# Додаємо хрестики ('mx'), щоб показати 5 штучних рівновіддалених вузлів
plt.plot(x_eq, y_eq, 'mx', markersize=8, label="Рівновіддалені вузли")

plt.plot([target_rps, target_rps],[cpu_4, cpu_5], 'yo', label="Прогнозовані вузли")
plt.plot([target_rps, target_rps],[cpu_4, cpu_fact], 'mo', label="Прогноз факторіальним многочленом")
plt.xlabel("RPS")
plt.ylabel("CPU (%)")
plt.title("Інтерполяція завантаження CPU")
plt.legend()
plt.grid()
plt.show()

print(f"Прогнозоване завантаження CPU при {target_rps} RPS (метод Ньютона):")
print(f" - 4 вузли: {cpu_4:.2f}%")
print(f" - 5 вузлів: {cpu_5:.2f}%")
print(f" - Факторіальним многочленом (5 рівновіддалених вузлів): {cpu_fact:.2f}%")
print(f"Різниця прогнозів (оцінка стабільності): {error:.2f}%")
print("-" * 30)
print(f"Значення функції вузлів w(x) у точці {target_rps}: {omega_val}")

max_error_v2 = np.max(error_v2)
mean_error_v2 = np.mean(error_v2)
print(f"Максимальна різниця між моделями на всьому проміжку: {max_error_v2:.2f}%")
print(f"Середня різниця між моделями на всьому проміжку: {mean_error_v2:.2f}%")
print("-" * 30)

plt.plot(x_vals_5, error_v2, 'm-', label="|Прогноз(5) - Прогноз(4)|")
plt.xlabel("RPS")
plt.ylabel("Абсолютна різниця CPU (%)")
plt.title("Оцінка стабільності моделі")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# -------------------------------
# Функція sin(x)
# -------------------------------
# Залежність інтерполяції від кроку h
# різна кількість вузлів, постійний інтервал
# -------------------------------
print()
print("п. 4: Різний крок h, постійний інтервал [a, b]")
a = 0
b = 6 * np.pi
n_list = [5, 10, 20]

print(f"Інтервал: [{a}, {b}]")

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Сітка для табуляції та малювання
h_plot = (b - a) / (20 * 20)
x_dense = np.arange(a, b + h_plot, h_plot)
y_true = np.sin(x_dense)

ax1.plot(x_dense, y_true, 'k-', linewidth=2, label="f(x) = sin(x)")

colors = ['b', 'g', 'r']

for i, n_sin in enumerate(n_list):
	# Створюємо вузли для моделі
	x_nodes = np.linspace(a, b, n_sin)
	y_nodes = np.sin(x_nodes)
	
	model_sin = NewtonInterpolator(x_nodes, y_nodes)

	y_approx = np.array([model_sin.newton_polynomial(xi) for xi in x_dense])
	errors = np.abs(y_true - y_approx)
	omega_vals = np.array([model_sin.omega_function(xi) for xi in x_dense])
	
	max_err = np.max(errors)
	mean_err = np.mean(errors)
	print(f"n = {n_sin} вузлів | Макс. похибка: {max_err:.4e} | Середня похибка: {mean_err:.4e}")
	
	ax1.plot(x_dense, y_approx, color=colors[i], linestyle='--', label=f"N(x) ({n_sin} вузлів)")
	ax1.plot(x_nodes, y_nodes, marker='o', color=colors[i], linestyle='None')

	ax2.plot(x_dense, errors, color=colors[i], label=f"Похибка ({n_sin} вузлів)")
	ax3.plot(x_dense, omega_vals, color=colors[i], label=f"w(x) ({n_sin} вузлів)")


ax1.set_title(f"Інтерполяція функції sin(x) на 5, 10 та 20 вузлах")
ax1.legend()
ax1.grid(True)

ax2.set_title("Похибка інтерполяції ε(x)")
ax2.legend()
ax2.grid(True)

ax3.set_title("Функція вузлів w_n(x)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Залежність інтерполяції від вузлів
# фіксований крок, змінний інтервал
# -------------------------------
print("п. 4: Постійний крок h, різний інтервал [a, b]")
a_var = 0
h_fixed = 1 #0.5

print(f"Початок інтервалу: a = {a_var}, крок = {h_fixed}")

fig_var, ax_var = plt.subplots(figsize=(10, 6))

# Реальна функція для найдовшого можливого інтервалу
b_max = a_var + h_fixed * 20
x_max_dense = np.arange(a_var, b_max + 0.01, 0.05)
ax_var.plot(x_max_dense, np.sin(x_max_dense), 'k-', linewidth=2, alpha=0.3, label="f(x) = sin(x)")

for i, n_val in enumerate(n_list):
    # Рахуємо кінець інтервалу за формулою
    b = a_var + h_fixed * n_val

    x_nodes_var = np.arange(a_var, b, h_fixed)

    x_nodes_var = x_nodes_var[:n_val] 
    y_nodes_var = np.sin(x_nodes_var)

    model_var = NewtonInterpolator(x_nodes_var, y_nodes_var)

    x_dense_var = np.linspace(a_var, x_nodes_var[-1], 100)
    y_approx_var = np.array([model_var.newton_polynomial(xi) for xi in x_dense_var])

    ax_var.plot(x_dense_var, y_approx_var, color=colors[i], linestyle='--', label=f"N(x) ({n_val} вузлів)")
    ax_var.plot(x_nodes_var, y_nodes_var, marker='o', color=colors[i], linestyle='None')

ax_var.set_title(f"Інтерполяція sin(x) (фіксований h={h_fixed})")
ax_var.set_xlabel("x")
ax_var.set_ylabel("y")
ax_var.legend()
ax_var.grid(True)

plt.tight_layout()
plt.show()