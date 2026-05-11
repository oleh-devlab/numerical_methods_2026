import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_rosenbrock_trajectory(trajectory):
    x1 = np.linspace(-1.5, 1.5, 400)
    x2 = np.linspace(-0.5, 1.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    Z = 100 * (X1**2 - X2)**2 + (X1 - 1)**2

    plt.figure(figsize=(10, 8))
    
    levels = np.logspace(-1, 3, 20)
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Значення функції Розенброка (лог. шкала)')

    traj_x = [point[0] for point in trajectory]
    traj_y = [point[1] for point in trajectory]

    plt.plot(traj_x, traj_y, marker='o', color='red', markersize=4, linestyle='-', linewidth=2, label='Траєкторія Хука-Дживса')
    
    plt.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Старт')
    plt.plot(1.0, 1.0, 'b*', markersize=12, label='Справжній мінімум (1, 1)')

    plt.title("Траєкторія пошуку мінімуму функції Розенброка")
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_system():
    x1 = np.linspace(-3, 3, 400)
    x2 = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    F1 = X1**2 + X2**2 - 4
    F2 = X2 - X1**2

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, F1, levels=[0], colors='blue')
    plt.contour(X1, X2, F2, levels=[0], colors='red')
    
    plt.plot([], [], color='blue', label='$x_1^2 + x_2^2 - 4 = 0$')
    plt.plot([], [], color='red', label='$x_2 - x_1^2 = 0$')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title("Графіки системи нелінійних рівнянь")
    plt.legend()
    plt.show()

class HookeJeevesOptimizer:
    def __init__(self, target_function, eps1=1e-5, eps2=1e-5, step_reduce=2.0, p=2.0):
        self.func = target_function
        self.eps1 = eps1
        self.eps2 = eps2
        self.q = step_reduce
        self.p = p

    def optimize(self, start_point, initial_step):
        x_base = np.array(start_point, dtype=float)
        step_sizes = np.array(initial_step, dtype=float)
        iterations = 0
        
        trajectory = [np.copy(x_base)]
        
        while True:
            iterations += 1
            x_new = np.copy(x_base)
            
            # Досліджуючий пошук
            for i in range(len(x_new)):
                while True:
                    test_point = np.copy(x_new)
                    
                    test_point[i] += step_sizes[i]
                    if self.func(test_point) < self.func(x_new):
                        x_new = test_point
                        break
                    
                    test_point[i] -= 2 * step_sizes[i] 
                    if self.func(test_point) < self.func(x_new):
                        x_new = test_point
                        break
                    
                    step_sizes[i] /= self.q
                    if step_sizes[i] < self.eps1:
                        break
            
            if not np.array_equal(x_new, x_base):
                norm_dx = np.linalg.norm(step_sizes)
                df = abs(self.func(x_new) - self.func(x_base))
                
                if norm_dx < self.eps1 and df < self.eps2:
                    x_base = x_new
                    trajectory.append(np.copy(x_base))
                    break
            else:
                # Якщо точка X^1 == X^0 і кроки стали < eps1, то це мінімум
                break
                
            # Пошук по зразку
            while True:
                x_pattern = x_new + self.p * (x_new - x_base)
                
                x_pattern_new = np.copy(x_pattern)
                for i in range(len(x_pattern_new)):
                    test_point = np.copy(x_pattern_new)
                    
                    test_point[i] += step_sizes[i]
                    if self.func(test_point) < self.func(x_pattern_new):
                        x_pattern_new = test_point
                    else:
                        test_point[i] -= 2 * step_sizes[i]
                        if self.func(test_point) < self.func(x_pattern_new):
                            x_pattern_new = test_point

                if self.func(x_pattern_new) < self.func(x_new):
                    x_base = np.copy(x_new)
                    x_new = np.copy(x_pattern_new)
                    trajectory.append(np.copy(x_new))
                else:
                    x_base = np.copy(x_new)
                    trajectory.append(np.copy(x_base))
                    break # Повертаємось до досліджуючого пошуку

        return x_base, iterations, trajectory

# --- Цільові функції ---
def rosenbrock_function(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

def objective_function_system(x):
    f1 = x[0]**2 + x[1]**2 - 4
    f2 = x[1] - x[0]**2
    return f1**2 + f2**2

def write_trajectory_to_file(trajectory, filename):
    filename = os.path.join(BASE_DIR, filename)
    with open(filename, 'w', encoding='utf-8') as f:
        for i, point in enumerate(trajectory):
            f.write(f"Крок {i}: x1={point[0]:.6f}, x2={point[1]:.6f}\n")

def main():
    print("--- Тестування на функції Розенброка ---")
    opt_ros = HookeJeevesOptimizer(rosenbrock_function, eps1=1e-6, eps2=1e-6)
    x0_ros = [-1.2, 0.0]
    step_ros = [0.5, 0.5]
    
    res_ros, iters_ros, traj_ros = opt_ros.optimize(x0_ros, step_ros)
    
    print(f"Початкове наближення: {x0_ros}")
    print(f"Знайдений мінімум: x* = [{res_ros[0]:.6f}, {res_ros[1]:.6f}] (Очікуваний: [1.0, 1.0])")
    print(f"Значення функції в точці мінімуму: {rosenbrock_function(res_ros):.10e}")
    print(f"Число ітерацій: {iters_ros}\n")

    print("--- Знаходження розв'язку системи нелінійних рівнянь ---")
    opt_sys = HookeJeevesOptimizer(objective_function_system, eps1=1e-6, eps2=1e-6)
    x0_sys = [1.2, 1.5] 
    initial_step_sys = [0.5, 0.5]
    
    res_sys, iters_sys, trajectory = opt_sys.optimize(x0_sys, initial_step_sys)
    
    print(f"Початкове наближення: {x0_sys}")
    print(f"Знайдений корінь: x* = [{res_sys[0]:.6f}, {res_sys[1]:.6f}]")
    print(f"Значення цільової функції Ф(X*): {objective_function_system(res_sys):.10e}")
    print(f"Число кроків (ітерацій) на траєкторії спуску: {iters_sys}")
    print(f"Загальна кількість точок траєкторії: {len(trajectory)}")
    
    write_trajectory_to_file(trajectory, "hooke_jeeves_trajectory.txt")
    print("Координати точок траєкторії збережено у 'hooke_jeeves_trajectory.txt'")

    plot_rosenbrock_trajectory(traj_ros)
    plot_system()

if __name__ == '__main__':
    main()