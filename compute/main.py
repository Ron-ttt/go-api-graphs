import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode, freqresp, step, impulse
import json
import os
import control as ctrl

# Функция для парсинга передаточной функции и извлечения коэффициентов
def parse_transfer_function(tf):
    tf = tf.replace(" ", "")
    match = re.match(r"\((.*?)\)/\((.*?)\)", tf)
    if not match:
        raise ValueError("Неправильный формат: должно быть (числитель)/(знаменатель)")

    num_expr = match.group(1)
    den_expr = match.group(2)

    num_coeffs = extract_coefficients(num_expr)
    den_coeffs = extract_coefficients(den_expr)

    return num_coeffs, den_coeffs

# Функция для извлечения коэффициентов из строки
def extract_coefficients(expr):
    expr = expr.replace(",", ".").replace("-", "+-")
    terms = expr.split("+")
    coeffs = {}
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
            
        # Обработка случая с пустым термином (например, из-за +-)
        if term.startswith("-"):
            term = term[1:]
            sign = -1
        else:
            sign = 1
            
        # Обработка каждого термина
        match = re.match(r"(\d*\.?\d*)\*?s\^(\d+)", term)
        if match:
            coeff = float(match.group(1) or 1) * sign
            degree = int(match.group(2))
            coeffs[degree] = coeff
        else:
            # Обработка термина вида 3*s, 5
            match = re.match(r"(-?\d*\.?\d*)\*?s", term)
            if match:
                coeff = float(match.group(1) if match.group(1) else 1)
                coeffs[1] = coeff
            else:
                # Константа
                match = re.match(r"(-?\d*\.?\d*)", term)
                if match:
                    coeff = float(match.group(1))
                    coeffs[0] = coeff

    # Ensure all degrees are included in the list
    max_degree = max(coeffs.keys(), default=0)
    ordered_coeffs = [coeffs.get(i, 0.0) for i in range(max_degree + 1)]
    
    return ordered_coeffs


def get_auto_omega(sys, n_points=500):
    """Автоматически определяет диапазон частот для графиков на основе полюсов и нулей системы."""
    poles = ctrl.poles(sys)
    zeros = ctrl.zeros(sys)
    
    # Убираем полюса/нули в нуле и бесконечности (если есть)
    finite_poles = poles[np.isfinite(poles)]
    finite_zeros = zeros[np.isfinite(zeros)]
    
    # Если нет полюсов/нулей (например, интегратор или чисто статическая система)
    if len(finite_poles) == 0 and len(finite_zeros) == 0:
        return np.logspace(-2, 2, n_points)  # Стандартный диапазон
    
    # Находим минимальную и максимальную "значимые" частоты (мнимая часть полюсов/нулей)
    all_critical = np.concatenate([finite_poles, finite_zeros])
    im_freqs = np.abs(np.imag(all_critical))
    
    # Убираем нулевые частоты (вещественные полюса/нули)
    im_freqs = im_freqs[im_freqs > 0]
    
    if len(im_freqs) == 0:
        # Все полюса/нули вещественные — берем модуль и добавляем запас
        min_freq = 0.1 * np.min(np.abs(all_critical))
        max_freq = 10 * np.max(np.abs(all_critical))
    else:
        # Есть комплексные полюса/нули — берем их частоты
        min_freq = 0.1 * np.min(im_freqs)
        max_freq = 10 * np.max(im_freqs)
    
    # Защита от слишком маленьких/больших частот
    min_freq = max(min_freq, 1e-3)  # Не ниже 0.001 рад/с
    max_freq = min(max_freq, 1e5)   # Не выше 100 000 рад/с
    
    return np.logspace(np.log10(min_freq), np.log10(max_freq), n_points)

# Функция для построения графиков и сохранения их
def plot_graphs(num_coeffs, den_coeffs, static_dir):
    # Передаточная функция
    tf_sys = ctrl.TransferFunction(num_coeffs, den_coeffs)

    poles = ctrl.poles(tf_sys)
    zeros = ctrl.zeros(tf_sys)

    # Автоматически определяем частоты
    omega = get_auto_omega(tf_sys)

    # Получаем частотную характеристику
    response = ctrl.frequency_response(tf_sys, omega)
    
    # Преобразуем массивы к нужной размерности
    mag = np.squeeze(response.magnitude)  # из (1,500) в (500,)
    phase = np.squeeze(response.phase)    # из (1,500) в (500,)
    omega = response.frequency           # уже (500,)
    
    # Вычисляем real и imag из magnitude и phase
    real = mag * np.cos(phase)
    imag = mag * np.sin(phase)

    # 1. График Боде
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(omega, 20 * np.log10(mag))
    plt.title('Частотная характеристика (Амплитуда)')
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogx(omega, np.degrees(phase))
    plt.title('Частотная характеристика (Фаза)')
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Фаза (градусы)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'bode.png'))
    plt.close()

    # 2. Переходная характеристика
    plt.figure()
    t, y = ctrl.step_response(tf_sys)
    plt.plot(t, y)
    plt.title('Переходная характеристика')
    plt.xlabel('Время [с]')
    plt.ylabel('Амплитуда')
    plt.savefig(os.path.join(static_dir, 'step_response.png'))
    plt.close()

    # 3. Импульсная характеристика
    plt.figure()
    t, y = ctrl.impulse_response(tf_sys)
    plt.plot(t, y)
    plt.title('Импульсная характеристика')
    plt.xlabel('Время [с]')
    plt.ylabel('Амплитуда')
    plt.savefig(os.path.join(static_dir, 'impulse_response.png'))
    plt.close()

    # 4. Годограф Найквиста
    plt.figure()
    plt.plot(real, imag, label='Годограф Найквиста')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Годограф Найквиста')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(static_dir, 'nyquist_plot.png'))
    plt.close()

    # 5. Годограф Михайлова
    plt.figure()
    plt.plot(real, -imag, label='Годограф Михайлова')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Годограф Михайлова')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(static_dir, 'mikhailov_plot.png'))
    plt.close()

    # 6. Нули и полюса
    plt.figure()
    plt.scatter(np.real(poles), np.imag(poles), marker='x', label="Полюса", color='red')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label="Нули", color='blue')
    plt.title('Нули и полюса')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(static_dir, 'poles_zeros.png'))
    plt.close()

    # Метрики (остается без изменений)
    zeros_str = ", ".join([f"{z.real:.2f}+{z.imag:.2f}j" for z in zeros])
    poles_str = ", ".join([f"{p.real:.2f}+{p.imag:.2f}j" for p in poles])

    try:
        critical_frequency_idx = np.where(np.degrees(phase) < -180)[0][0]
        phase_at_critical_freq = np.degrees(phase)[critical_frequency_idx]
        stability_margin = 180 + phase_at_critical_freq
    except IndexError:
        stability_margin = None

    settling_time = np.max(t)
    overshoot = np.max(y)

    return {
        "bode": "/static/bode.png",
        "step_response": "/static/step_response.png",
        "impulse_response": "/static/impulse_response.png",
        "nyquist_plot": "/static/nyquist_plot.png",
        "mikhailov_plot": "/static/mikhailov_plot.png",
        "poles_zeros": "/static/poles_zeros.png",
        "zeros": zeros_str,
        "poles": poles_str,
        "settling_time": float(settling_time),
        "overshoot": float(overshoot),
        "stability_margin": float(stability_margin) if stability_margin is not None else None
    }
# Основная логика
def log(*args, **kwargs):
    """Функция для логирования в stderr"""
    print(*args, file=sys.stderr, **kwargs)

if __name__ == "__main__":
    tf = sys.argv[1]
    static_dir = sys.argv[2]

    log(f"Получена передаточная функция: {tf}")

    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf)
        log(f"Числитель: {num_coeffs}")
        log(f"Знаменатель: {den_coeffs}")

        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        response = plot_graphs(num_coeffs, den_coeffs, static_dir)
        # Выводим только JSON в stdout
        print(json.dumps(response))

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        print(json.dumps({"status": "error", "message": error_msg}))
        sys.exit(1)