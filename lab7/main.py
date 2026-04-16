import matplotlib.pyplot as plt
import os

from RW_matrix import read_matrix, read_vector, generate_system, BASE_DIR

n = 100

def vector_norm(V):
    return max(abs(v) for v in V)

def matrix_norm(A):
    return max(sum(abs(x) for x in row) for row in A)

def main():
    if not os.path.exists(f"{BASE_DIR}/matrix_A.txt") or not os.path.exists(f"{BASE_DIR}/vector_B.txt"):
        print("Генерація нових даних...")
        generate_system(n)

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")

    eps_0 = 1e-14

    

    # Побудова графіка для порівняння швидкості збіжності
    plt.figure(figsize=(10, 6))

    plt.yscale('log')
    plt.title("Порівняння збіжності ітераційних методів")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма різниці ||dX|| (логарифмічна шкала)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.savefig(f'{BASE_DIR}/convergence_comparison_lab8.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()