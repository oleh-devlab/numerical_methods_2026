import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_system(n=100, x_val=2.5):
    A = np.random.uniform(1.0, 10.0, (n, n))
    np.fill_diagonal(A, A.diagonal() + 100.0)

    # Обчислення вектора B
    B = []
    for i in range(n):
        b_i = sum(A[i][j] * x_val for j in range(n))
        B.append(b_i)

    # Запис матриці А та вектора В у текстові файли
    with open(f"{BASE_DIR}/matrix_A.txt", "w") as fA:
        for row in A:
            fA.write(" ".join(str(x) for x in row) + "\n")

    with open(f"{BASE_DIR}/vector_B.txt", "w") as fB:
        for b in B:
            fB.write(str(b) + "\n")

    return A, B

def read_matrix(filename):
    filename = f"{BASE_DIR}/{filename}"
    A = []
    with open(filename, "r") as f:
        for line in f:
            A.append([float(x) for x in line.split()])
    return A

def read_vector(filename):
    filename = f"{BASE_DIR}/{filename}"
    B = []
    with open(filename, "r") as f:
        for line in f:
            B.append(float(line.strip()))
    return B

def save_lu(L, U, filename):
    filename = f"{BASE_DIR}/{filename}"
    with open(filename, "w") as f:
        f.write("Matrix L:\n")
        for row in L:
            f.write(" ".join(str(x) for x in row) + "\n")
        f.write("\nMatrix U:\n")
        for row in U:
            f.write(" ".join(str(x) for x in row) + "\n")

def main():
    generate_system()

if __name__ == "__main__":
    main()