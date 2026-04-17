import matplotlib.pyplot as plt
import os

from RW_matrix import read_matrix, read_vector, generate_system, BASE_DIR

n = 100

def multiply_matrix_vector(A, X):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(n))
    return res

def vector_norm(V):
    return max(abs(v) for v in V)

def matrix_norm(A):
    return max(sum(abs(x) for x in row) for row in A)

def simple_iteration(A, B, eps_0):
    n = len(A)
    X = [1.0] * n
    
    tau = 1.0 / matrix_norm(A) 
    
    iterations = 0
    history_dX = []

    while True:
        X_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n))
            # x^(k+1) = x^(k) - tau * A * x^(k) + tau * f
            X_new[i] = X[i] - tau * s + tau * B[i]

        dX = [X_new[i] - X[i] for i in range(n)]
        norm_dX = vector_norm(dX)
        history_dX.append(norm_dX)

        X = X_new[:]
        iterations += 1

        if norm_dX < eps_0:
            break
        if iterations > 2500:
            print("Перевищено ліміт ітерацій у методі простої ітерації")
            break

    return iterations, X, history_dX

def jacobi(A, B, eps_0):
    n = len(A)
    X = [1.0] * n
    iterations = 0
    history_dX = []

    while True:
        X_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if j != i)
            X_new[i] = (B[i] - s) / A[i][i]

        dX = [X_new[i] - X[i] for i in range(n)]
        norm_dX = vector_norm(dX)
        history_dX.append(norm_dX)

        X = X_new[:]
        iterations += 1

        if norm_dX < eps_0:
            break
        if iterations > 2500:
            print("Перевищено ліміт ітерацій у методі Якобі")
            break

    return iterations, X, history_dX

def seidel(A, B, eps_0):
    n = len(A)
    X = [1.0] * n
    iterations = 0
    history_dX = []

    while True:
        X_new = [0.0] * n
        for i in range(n):
            # Використовуємо вже знайдені нові значення X_new для j < i
            s1 = sum(A[i][j] * X_new[j] for j in range(i))
            # Використовуємо старі значення X для j > i
            s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
            X_new[i] = (B[i] - s1 - s2) / A[i][i]

        dX = [X_new[i] - X[i] for i in range(n)]
        norm_dX = vector_norm(dX)
        history_dX.append(norm_dX)

        X = X_new[:]
        iterations += 1

        if norm_dX < eps_0:
            break
        if iterations > 2500:
            print("Перевищено ліміт ітерацій у методі Зейделя")
            break

    return iterations, X, history_dX

def main():
    if not os.path.exists(f"{BASE_DIR}/matrix_A.txt") or not os.path.exists(f"{BASE_DIR}/vector_B.txt"):
        print("Генерація нових даних...")
        generate_system(n)

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    eps_0 = 1e-14
    print(f"{eps_0=}\n")

    # 1. Метод простої ітерації
    iter_simp, _, hist_simp = simple_iteration(A, B, eps_0)
    print(f"Метод простої ітерації завершено за {iter_simp} ітерацій.")

    # 2. Метод Якобі
    iter_jac, _, hist_jac = jacobi(A, B, eps_0)
    print(f"Метод Якобі завершено за {iter_jac} ітерацій.")

    # 3. Метод Зейделя
    iter_seid, X_seid, hist_seid = seidel(A, B, eps_0)
    print(f"Метод Зейделя завершено за {iter_seid} ітерацій.")
    
    print("\nПерші 3 елементи розв'язку (метод Зейделя):")
    for i in range(3):
        print(f"x[{i}] = {X_seid[i]:.6f}")

    # Побудова графіка для порівняння швидкості збіжності
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(hist_simp) + 1), hist_simp, label='Проста ітерація', color='blue', alpha=0.7)
    plt.plot(range(1, len(hist_jac) + 1), hist_jac, label='Метод Якобі', color='green', alpha=0.7)
    plt.plot(range(1, len(hist_seid) + 1), hist_seid, label='Метод Зейделя', color='red', alpha=0.7)

    plt.yscale('log')
    plt.title("Порівняння збіжності ітераційних методів")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма різниці ||dX|| (логарифмічна шкала)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig(f'{BASE_DIR}/convergence_comparison_lab7.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збіжності збережено як 'convergence_comparison_lab7.png'")
    plt.show()

if __name__ == '__main__':
    main()