import numpy as np
import matplotlib.pyplot as plt

from tabulate_data import read_data

class LeastSquaresApproximator:
	def __init__(self, x, y, degree = None, max_degree = 10):
		self.x_data = x
		self.y = y
		self.max_degree = max_degree

		if degree is None:
			self.m = self._get_optimal_m()
		else:
			self.m = degree

		self.A = self._form_matrix(self.m)
		self.b_vec = self._form_vector(self.m)
		self.coef = self._gauss_solve()

		if degree is not None:
			self.variances = [self.variance(self.polynomial(self.x_data))]

	# -------------------------------
	# 3. Вибір оптимального ступеня полінома
	# -------------------------------
	def _get_optimal_m(self):
		self.variances = []

		for m in range(1, self.max_degree + 1):
			self.A = self._form_matrix(m)
			self.b_vec = self._form_vector(m)
			self.coef = self._gauss_solve()
			y_approx = self.polynomial(self.x_data)
			var = self.variance(y_approx)
			self.variances.append(var)

		return (np.argmin(self.variances) + 1)

	# -------------------------------
	# 2. Функції МНК
	# -------------------------------
	def _form_matrix(self, m):
		A = np.zeros((m+1, m+1))
		for i in range(m+1):
			for j in range(m+1):
				A[i, j] = np.sum(self.x_data ** (i + j))
		return A

	def _form_vector(self, m):
		b = np.zeros(m+1)
		for i in range(m+1):
			b[i] = np.sum(self.y * (self.x_data ** i))
		return b

	def _gauss_solve(self):
		A = self.A.copy()
		b = self.b_vec.copy()
		n = len(b)

		# Прямий хід з вибором головного елемента
		for k in range(n-1):
			max_row = np.argmax(np.abs(A[k:, k])) + k
			A[[k, max_row]] = A[[max_row, k]]
			b[[k, max_row]] = b[[max_row, k]]
			
			for i in range(k+1, n):
				factor = A[i, k] / A[k, k]
				A[i, k:] -= factor * A[k, k:]
				b[i] -= factor * b[k]

		# Зворотній хід
		x_sol = np.zeros(n)
		for i in range(n-1, -1, -1):
			x_sol[i] = (b[i] - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]
		return x_sol

	def polynomial(self, x):
		y_poly = np.zeros_like(x, dtype=float)
		for i in range(len(self.coef)):
			y_poly += self.coef[i] * (x ** i)
		return y_poly

	def variance(self, y_approx):
		return np.mean((self.y - y_approx) ** 2)
	
	def get_graph_vals(self):
		x_vals = np.linspace(self.x_data.min(), self.x_data.max(), 20*len(self.x_data) + 1)
		y_vals = self.polynomial(x_vals)

		return np.array(x_vals), np.array(y_vals)

def calculate_error(y_true, y_approx):
	return np.abs(y_true - y_approx)

# -------------------------------
# 1. Вхідні дані
# -------------------------------
x, y = read_data("data.csv")
x = np.array(x)
y = np.array(y)

# -------------------------------
# 4. Побудова апроксимації
# -------------------------------
model = LeastSquaresApproximator(x, y, max_degree = 10)
y_approx = model.polynomial(x)

# -------------------------------
# 5. Прогноз на наступні 3 місяці
# -------------------------------
x_future = np.array([25, 26, 27])
y_future = model.polynomial(x_future)

# -------------------------------
# 7. Вивід результатів
# -------------------------------
print("\nДисперсії для різних степенів:")
for i, var in enumerate(model.variances):
    print(f"m={i+1}: {var:.4f}")

print(f"Оптимальний степінь полінома: {model.m}")

print("\nПрогноз температури:")
for mnth, temp in zip(x_future, y_future):
    print(f"Місяць {mnth}: {temp:.1f} °C")

x_vals, y_vals = model.get_graph_vals()

# Побудова графіка апроксимації
plt.plot(x, y, 'ro', label="Вузли")
plt.plot(x_vals, y_vals, 'b-', label=f"Поліном (m={model.m})")
# plt.plot(x_future, y_future, 'r^', markersize=8, label="Прогноз (екстраполяція)")
plt.xlabel("Місяці")
plt.ylabel("Температура")
plt.title("Апроксимація МНК")
plt.legend()
plt.grid()
plt.show()

# Побудова графіка дисперсії
plt.plot(range(1, model.max_degree + 1), model.variances, 'ro-')
plt.axvline(model.m, color='r', linestyle='--', label=f'Оптимальне m={model.m}')
plt.xlabel("Степінь")
plt.ylabel("Дисперсія")
plt.title("Залежність величини дисперсії від степені апроксимуючого многочлена")
plt.legend()
plt.grid()
plt.show()

# Побудова графіків похибки для m = 1...10
plt.figure(figsize=(10, 6))

n = len(x)
h1 = (x.max() - x.min()) / (20 * n)
x_err_grid = np.arange(x.min(), x.max(), h1)

# За еталонну функцію взято апроксимацію з найвищим степенем
reference_model = LeastSquaresApproximator(x, y, degree=10)
y_true_reference = reference_model.polynomial(x_err_grid)

colors = plt.cm.tab10.colors # список з 10 кольорів

for m in range(1, 11):
    temp_model = LeastSquaresApproximator(x, y, degree=m)
    y_poly_grid = temp_model.polynomial(x_err_grid)
    
    error_vals = calculate_error(y_true_reference, y_poly_grid)
    
    line_style = '-' if m <= 5 else '--'
    plt.plot(x_err_grid, error_vals, linestyle=line_style, color=colors[m-1], label=f"m={m}")

plt.xlabel("Місяці")
plt.ylabel("Модуль похибки |f(x) - φ(x)|")
plt.title("Функції похибки для різних степенів")
plt.grid()

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()