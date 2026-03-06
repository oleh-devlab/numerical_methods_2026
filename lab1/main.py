import requests
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------
# 1. Запит до Open-Elevation API
# -------------------------------

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()

results = data["results"]
n = len(results)

# Вивід таблиці вузлів
print(f"Кількість вузлів: {n}")
print("\nТабуляція вузлів:")
print(" № | Latitude  | Longitude | Elevation (m)")

with open(os.path.join(BASE_DIR, "elevation_data.txt"), "w", encoding="utf-8") as f:
    f.write(f"Кількість вузлів: {n}\n")
    f.write("Табуляція вузлів:\n")
    f.write(" № | Latitude  | Longitude | Elevation (m)\n")
    for i, point in enumerate(results):
        line = (f"{i:2d} | {point['latitude']:.6f} | "
                f"{point['longitude']:.6f} | "
                f"{point['elevation']:.2f}\n")
        f.write(line)
        print(line, end="")


# -------------------------------
# 2. Кумулятивна відстань
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(
        lat2)  # Переводимо широти у радіани
    dphi = np.radians(lat2 - lat1)  # Різниця широт у радіанах
    dlambda = np.radians(lon2 - lon1)  # Різниця довгот у радіанах
    a = np.sin(
        dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # Відстань у метрах


# Створюємо списки координат і висот
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

# Обчислюємо кумулятивну відстань
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

# Вивід таблиці відстаней
print("\nТабуляція (відстань, висота):")
print(" № | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

# Запис табуляції у текстовий файл
with open(os.path.join(BASE_DIR, "tabulation.txt"), "w", encoding="utf-8") as f:
    f.write("№ | Latitude | Longitude | Elevation (m) | Distance (m)\n")
    for i, point in enumerate(results):
        f.write(f"{i:2d} | {point['latitude']:.6f} | "
                f"{point['longitude']:.6f} | "
                f"{point['elevation']:.2f} | "
                f"{distances[i]:.2f}\n")


# -------------------------------
# 3. Кубічний сплайн (Математика)
# -------------------------------
def build_tridiagonal_system(n_nodes, h, x, y):
    A = np.zeros(n_nodes)
    B = np.zeros(n_nodes)
    C = np.zeros(n_nodes)
    D = np.zeros(n_nodes)

    B[0] = 1
    B[-1] = 1

    for i in range(1, n_nodes - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    return A, B, C, D


def solve_tridiagonal(n_nodes, A, B, C, D):
    # Алгоритм Томаса
    # Пряма прогонка
    for i in range(1, n_nodes):
        m = A[i] / B[i - 1]
        B[i] = B[i] - m * C[i - 1]
        D[i] = D[i] - m * D[i - 1]

    M = np.zeros(n_nodes)
    M[-1] = D[-1] / B[-1]

    # Зворотна прогонка
    for i in range(n_nodes - 2, -1, -1):
        M[i] = (D[i] - C[i] * M[i + 1]) / B[i]

    return M


def compute_spline_coefficients(n_nodes, h, x, y, M):
    a = y[:-1]
    b = np.zeros(n_nodes - 1)
    c = M[:-1] / 2
    d = np.zeros(n_nodes - 1)

    for i in range(n_nodes - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        d[i] = (M[i + 1] - M[i]) / (6 * h[i])

    return a, b, c, d


def calculate_splines(x, y, print_math=True):
    n_nodes = len(x)
    h = np.diff(x)

    A, B, C, D = build_tridiagonal_system(n_nodes, h, x, y)
    M = solve_tridiagonal(n_nodes, A, B, C, D)
    a, b, c, d = compute_spline_coefficients(n_nodes, h, x, y, M)

    # Виводимо інформацію лише для 21 вузла
    if print_math:
        print("\n--- ПРОМІЖНІ РОЗРАХУНКИ (Пункти 6-9) ---")
        print("Інтервали h:")
        print(h)

        # п. 6-9
        
        print("\nП. 6. Коефіцієнти системи рівнянь:")
        for i in range(1, n_nodes - 1):
            print(f"Вузол {i}: A={A[i]:.2f}, B={B[i]:.2f}, C={C[i]:.2f}, D={D[i]:.4f}")
        
        print("\nП. 7. Результат методу прогонки M:")
        for i in range(n_nodes):
            print(f"M[{i}] = {M[i]:.6f}")

        print("\nП. 8 та 9. Коефіцієнти сплайнів:")
        for i in range(n_nodes - 1):
            print(f"Інтервал {i}: a={a[i]:.2f}, b={b[i]:.4f}, c={c[i]:.6f}, d={d[i]:.6f}")
        print("----------------------------------------------------------------\n")

    return a, b, c, d, x


def spline_eval(xi, a, b, c, d, x_nodes):
    N = len(x_nodes) - 1
    for i in range(N):
        if x_nodes[i] <= xi <= x_nodes[i + 1] or (i == N - 1
                                                  and xi >= x_nodes[-1]):
            dx = xi - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return None


# -------------------------------
# Аналіз та графіки
# -------------------------------

x_full = np.array(distances)
y_full = np.array(elevations)

# Рахуємо сплайни для всіх точок
a_full, b_full, c_full, d_full, x_nodes_full = calculate_splines(
    x_full, y_full, print_math=True)

xx = np.linspace(x_full[0], x_full[-1], 1000)
yy_full = np.array([
    spline_eval(xi, a_full, b_full, c_full, d_full, x_nodes_full) for xi in xx
])

# Функція для вибору k вузлів, побудови сплайна і розрахунку похибки
def test_nodes(k, x_full, y_full, xx, yy_full):
    indices = np.linspace(0, len(x_full) - 1, k, dtype=int)
    x_k = x_full[indices]
    y_k = y_full[indices]

    # Вимикаємо вивід математики для цих тестів, щоб консоль була чистою
    a_k, b_k, c_k, d_k, x_nodes_k = calculate_splines(x_k, y_k, print_math=False)

    yy_k = np.array(
        [spline_eval(xi, a_k, b_k, c_k, d_k, x_nodes_k) for xi in xx])
    error = np.abs(yy_full - yy_k)

    print(f"==== {k} вузлів ====")
    print(f"Максимальна похибка: {np.max(error)}")
    print(f"Середня похибка: {np.mean(error)}\n")

    return yy_k, error


yy_10, err_10 = test_nodes(10, x_full, y_full, xx, yy_full)
yy_15, err_15 = test_nodes(15, x_full, y_full, xx, yy_full)
yy_20, err_20 = test_nodes(20, x_full, y_full, xx, yy_full)

# Побудова графіків похибок та профілю
plt.figure(figsize=(10, 5))
plt.plot(xx,
         yy_full,
         label="21 вузол (еталон)",
         color="black",
         linewidth=2)
plt.plot(xx, yy_10, label="10 вузлів", linestyle="--")
plt.plot(xx, yy_15, label="15 вузлів", linestyle="-.")
plt.plot(xx, yy_20, label="20 вузлів", linestyle=":")
plt.legend()
plt.title("Вплив кількості вузлів")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(xx, err_10, label="10 вузлів")
plt.plot(xx, err_15, label="15 вузлів")
plt.plot(xx, err_20, label="20 вузлів")
plt.legend()
plt.title("Похибка апроксимації")
plt.xlabel("Відстань (м)")
plt.ylabel("Похибка (м)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, 'o-', color='green', label='Вузли (GPS дані)')
plt.plot(xx, yy_full, '-', color='red', alpha=0.6, label='Кубічний сплайн')
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти маршруту: ст. Заросляк - г. Говерла")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Додаткові характеристики
# -------------------------------

total_ascent = sum(
    max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(
    max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))

print("--- Додаткові характеристики ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

# Аналіз та графік градієнта
grad_full = np.gradient(yy_full, xx) * 100
steep_sections = np.where(np.abs(grad_full) > 15)[0]

print("\n--- Аналіз градієнта ---")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")
print(f"Кількість ділянок з крутизною > 15%: {len(steep_sections)}")

plt.figure(figsize=(10, 5))
plt.plot(xx, grad_full, color='purple')
plt.title("Градієнт маршруту (%)")
plt.xlabel("Відстань (м)")
plt.ylabel("Градієнт (%)")
plt.grid(True)
plt.show()

# Кумулятивна енергія (Масив та графік)
mass = 80
g = 9.81
energy_arr = [0]

# Накопичення енергії для кожної точки графіка (за сплайном)
for i in range(1, len(xx)):
    dh = yy_full[i] - yy_full[i - 1]
    if dh > 0:
        energy_arr.append(energy_arr[-1] + mass * g * dh)
    else:
        energy_arr.append(energy_arr[-1])

energy_arr = np.array(energy_arr)

# Енергія розрахована за дискретними вузлами
energy_discrete = mass * g * total_ascent

print("\n- Механічна робота -")
print("1. Розрахунок за дискретними GPS-точками:")
print(f"Механічна робота (Дж): {energy_discrete:.2f}")
print(f"Енергія (ккал): {energy_discrete / 4184:.2f}")

print("\n2. Розрахунок за інтерпольованим сплайном:")
print(f"Механічна робота (Дж) останнє значення: {energy_arr[-1]:.2f}")
print(f"Енергія (ккал): {energy_arr[-1] / 4184:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(xx, energy_arr, color='orange')
plt.title("Кумулятивна енергія (Дж)")
plt.xlabel("Відстань (м)")
plt.ylabel("Енергія (Дж)")
plt.grid(True)
plt.show()