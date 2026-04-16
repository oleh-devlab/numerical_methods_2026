from RW_matrix import read_matrix, read_vector, save_lu, generate_system
import os

n = 100

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            s1 = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - s1
            
        for j in range(k + 1, n):
            s2 = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = (A[k][j] - s2) / L[k][k]
            
    return L, U

def solve_lu(L, U, B):
    n = len(B)
    Z = [0.0] * n
    X = [0.0] * n
    
    # Розв'язок системи LZ = B
    for i in range(n):
        s1 = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - s1) / L[i][i]

    # Розв'язок системи UX = Z
    for i in range(n - 1, -1, -1):
        s2 = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s2
        
    return X

def multiply_matrix_vector(A, X):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(n))
    return res

def vector_norm(V):
    return max(abs(v) for v in V)

def refinement(X0, A, B, L, U, eps_0):
    iterations = 0
    X_current = X0[:]

    while True:
        # R = B - B(0)
        AX_current = multiply_matrix_vector(A, X_current)
        R = [B[i] - AX_current[i] for i in range(n)]

        dX = solve_lu(L, U, R)

        for i in range(n):
            X_current[i] += dX[i]

        iterations += 1

        norm_dX = vector_norm(dX)
        norm_R = vector_norm(R)

        if norm_dX <= eps_0 and norm_R <= eps_0:
            break
            
        if iterations > 50:
            print("Перервано: досягнуто ліміт ітерацій.")
            break

    return iterations, norm_dX, norm_R, X_current

def main():
    if not os.path.exists("matrix_A.txt") or not os.path.exists("vector_B.txt"):
        generate_system(n)

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    # Знаходження LU-розкладу
    L, U = lu_decomposition(A)
    save_lu(L, U, "lu_decomposition.txt")

    X0 = solve_lu(L, U, B)

    # 4. Оцінка точності знайденого розв'язку
    AX = multiply_matrix_vector(A, X0)
    diff = [AX[i] - B[i] for i in range(n)]
    eps = vector_norm(diff)
    print(f"Початкова похибка (eps): {eps}")

    # 5. Ітераційне уточнення розв'язку
    eps_0 = 1e-14

    iterations, norm_dX, norm_R, X_current = refinement(X0, A, B, L, U, eps_0)

    print(f"Уточнений розв'язок знайдено за {iterations} ітерацій.")
    print(f"Кінцева похибка: ||dX|| = {norm_dX}, ||R|| = {norm_R}")
    print("Перші 5 елементів уточненого розв'язку:")
    for i in range(5):
        print(f"x[{i}] = {X_current[i]}")

if __name__ == '__main__':
    main()