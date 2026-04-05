import math

def moisture(t):
    return 50.0 * math.exp(-0.1 * t) + 5.0 * math.sin(t)

def moisture_derivative_exact(t):
    return -5.0 * math.exp(-0.1 * t) + 5.0 * math.cos(t)

def central_difference(f, t0, h):
    return (f(t0 + h) - f(t0 - h)) / (2.0 * h)

def runge_romberg_refinement(d_h, d_2h):
    return d_h + ((d_h - d_2h) / 3)

def aitken_refinement(d_h, d_2h, d_4h):
    denominator = 2.0 * d_2h - (d_4h + d_h)
    if denominator == 0.0:
        return None
    return (d_2h**2 - d_4h * d_h) / denominator

def aitken_order(d_h, d_2h, d_4h):
    denominator = d_2h - d_h
    numerator = d_4h - d_2h
    
    if denominator == 0.0:
        return None
        
    ratio = abs(numerator / denominator)
    if ratio == 0.0:
        return None
        
    return math.log(ratio, 2.0)

def find_optimal_h(t0, exact):
    best_k = -3
    best_h = 10.0 ** (-best_k)
    best_value = central_difference(moisture, t0, best_h)
    best_error = abs(best_value - exact)

    print("\n- Пошук оптимального кроку -")
    for k in range(-3, 21):
        h = 10.0 ** (-k)
        approx = central_difference(moisture, t0, h)
        error = abs(approx - exact)
        print(f"h = 10^({-k}) = {h:.12e}, D(h) = {approx:.12f}, похибка = {error:.12e}")

        if error < best_error:
            best_h = h
            best_value = approx
            best_error = error
            best_k = k

    return best_h, best_value, best_error, best_k


t0 = 1
h = 1e-3

print("\n--- Вхідні дані ---")
print(f"Точка: t0 = {t0}")
print(f"Базовий крок: h = {h}")

exact = moisture_derivative_exact(t0)
print("\n--- 1) Точне значення похідної ---")
print(f"M'(t0) = {exact:.12f}")

print("\n--- 2) Оптимальний крок ---")
h_opt, d_opt, err_opt, k_opt = find_optimal_h(t0, exact)
print(f"Оптимальний крок: h_opt = 10^({-k_opt}) = {h_opt}")
print(f"Похідна D(h_opt) = {d_opt:.12f}")
print(f"Похибка: {err_opt:.12e}")

print("\n--- 3-6) Розрахунки для h, 2h та метод Рунге-Ромберга ---")
d_h = central_difference(moisture, t0, h)
d_2h = central_difference(moisture, t0, 2.0 * h)

err_h = abs(d_h - exact)
err_2h = abs(d_2h - exact)

print(f"D(h)  = {d_h:.12f}")
print(f"D(2h) = {d_2h:.12f}")
print(f"Похибка D(h)  = {err_h:.12e}")
print(f"Похибка D(2h) = {err_2h:.12e}")

d_rr = runge_romberg_refinement(d_h, d_2h)
err_rr = abs(d_rr - exact)

print(f"\nМетод Рунге-Ромберга: D_RR = {d_rr:.12f}")
print(f"Похибка D_RR = {err_rr:.12e}")

print("\n--- Характер зміни похибки ---")
if err_h != 0:
    step_ratio = err_2h / err_h
    print(f"- При зменшенні кроку вдвічі похибка базового методу зменшилась у {step_ratio:.2f} разів.")

if err_rr != 0 and err_h != 0:
    rr_improvement = err_h / err_rr
    print(f"- Застосування методу Рунге-Ромберга дозволило додатково зменшити похибку у {rr_improvement:.2f} разів порівняно з найкращим базовим значенням D(h).")

print("\n--- 7) Метод Ейткена (h, 2h, 4h) ---")
d_4h = central_difference(moisture, t0, 4.0 * h)
err_4h = abs(d_4h - exact)

print(f"D(h)  = {d_h:.12f}")
print(f"D(2h) = {d_2h:.12f}")
print(f"D(4h) = {d_4h:.12f}")
print(f"Похибка D(4h) = {err_4h:.12e}")

p_aitken = aitken_order(d_h, d_2h, d_4h)
d_aitken = aitken_refinement(d_h, d_2h, d_4h)

if p_aitken == None or d_aitken == None:
    print("\nПомилка: неможливо обчислити метод Ейткена (ділення на нуль).")
else:
    err_aitken = abs(d_aitken - exact)
    print(f"\nПорядок точності (Ейткен) p = {p_aitken:.8f}")
    print(f"Уточнене значення D_E = {d_aitken:.12f}")
    print(f"Похибка D_E = {err_aitken:.12e}")
    
    print("\n--- Характер зміни похибки (Ейткен) ---")
    if err_aitken != 0 and err_h != 0:
        aitken_improvement = err_h / err_aitken
        print(f"Застосування методу Ейткена зменшило похибку у {aitken_improvement:.2f} разів порівняно з базовим D(h).")
        print(f"Обчислений порядок точності p = {round(p_aitken)}.")

if exact < 0:
    print(f"\nM'(1) < 0: у точці t={t0} швидкість зміни вологості є від'ємною (ґрунт  втрачає вологу).")
    print("Оптимальний режим поливу: якщо при цьому абсолютна вологість M(t) наближається до мінімуму, увімкнути полив.")
else:
    if exact > 0:
        print(f"\nM'(1) > 0: у точці t={t0} вологість зростає.")
    else:
        print(f"\nM'(1) = 0: у точці t={t0} швидкість зміни вологості нульова (точка стабільності або локального екстремуму).")
    print("Оптимальний режим поливу: полив не потрібен.")