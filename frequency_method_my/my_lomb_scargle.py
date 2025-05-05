import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

# Создаем папку для результатов
if not os.path.exists('results'):
    os.makedirs('results')

def lomb_scargle_power(t, y, frequencies):
    """Реализация периодограммы Ломба-Скаргла с обработкой краевых случаев"""
    if len(t) < 2 or len(y) < 2 or len(frequencies) == 0:
        return np.zeros_like(frequencies)
    
    y_centered = y - np.mean(y)
    power = np.zeros_like(frequencies)
    
    for i, f in enumerate(frequencies):
        if f <= 0:
            continue
            
        # Вычисление сдвига τ
        angle = 4 * np.pi * f * t
        sum_sin = np.sum(np.sin(angle))
        sum_cos = np.sum(np.cos(angle))
        
        theta = 0.5 * np.arctan2(sum_sin, sum_cos)
        tau = theta / (2 * np.pi * f)
        
        # Вычисление компонент
        arg = 2 * np.pi * f * (t - tau)
        cos_arg = np.cos(arg)
        sin_arg = np.sin(arg)
        
        # Числители и знаменатели
        sum_cos_term = np.sum(y_centered * cos_arg)
        sum_sin_term = np.sum(y_centered * sin_arg)
        sum_cos2 = np.sum(cos_arg**2)
        sum_sin2 = np.sum(sin_arg**2)
        
        # Защита от деления на ноль
        sum_cos2 = max(sum_cos2, 1e-12)
        sum_sin2 = max(sum_sin2, 1e-12)
        
        power[i] = 0.5 * ((sum_cos_term**2 / sum_cos2) + 
                        (sum_sin_term**2 / sum_sin2))
            
    return power

# Диапазон пропущенных данных (0-90%)
precentage_of_missed_data = np.linspace(0., 0.9, 10)

for p in precentage_of_missed_data:
    # Генерация данных
    np.random.seed(42)
    t_full = np.linspace(0, 20, 500)
    y_true = (
        0.1 * t_full + 
        3 * np.sin(2*np.pi*0.8*t_full) + 
        2 * np.sin(2*np.pi*1.5*t_full) + 
        1.5 * np.sin(2*np.pi*0.3*t_full)
    )
    y_obs = y_true + np.random.normal(0, 0.3, len(t_full))
    
    # Разделение данных с проверкой
    split_idx = int(len(t_full)*0.8)
    train_mask = np.full(len(t_full), False)
    
    # Гарантируем минимальное количество точек
    min_train_points = 4
    max_attempts = 100
    attempt = 0
    
    while attempt < max_attempts:
        train_mask[:split_idx] = np.random.choice([True, False], split_idx, p=[1-p, p])
        if np.sum(train_mask) >= min_train_points:
            break
        attempt += 1
    else:
        print(f"Пропуск p={p:.1f}: не удалось получить минимальные {min_train_points} точек")
        continue
        
    test_mask = np.full(len(t_full), False)
    test_mask[split_idx:] = True

    t_train = t_full[train_mask]
    y_train = y_obs[train_mask]
    t_test = t_full[test_mask]
    y_test = y_obs[test_mask]

    if len(y_train) < min_train_points:
        print(f"Пропуск p={p:.1f}: осталось только {len(y_train)} точек")
        continue

    # Вычисление периодограммы
    freq = np.linspace(0.1, 10.0, 5000)
    try:
        power = lomb_scargle_power(t_train, y_train, freq)
    except Exception as e:
        print(f"Ошибка в периодограмме для p={p:.1f}: {str(e)}")
        continue
    
    # Поиск пиков
    if len(power) == 0 or np.all(power == 0):
        print(f"Пропуск p={p:.1f}: нулевая мощность в периодограмме")
        continue
        
    try:
        peaks, _ = find_peaks(power, prominence=np.std(power), distance=5)
    except Exception as e:
        print(f"Ошибка поиска пиков для p={p:.1f}: {str(e)}")
        continue
    
    if len(peaks) == 0:
        print(f"Пропуск p={p:.1f}: не найдены пики")
        continue
    
    try:
        top_freqs = freq[peaks][np.argsort(power[peaks])[-3:]]
    except IndexError:
        print(f"Пропуск p={p:.1f}: проблемы с выделением частот")
        continue

    if len(top_freqs) == 0:
        print(f"Пропуск p={p:.1f}: нет подходящих частот")
        continue
    
    # Определение модели
    def multi_harmonic_model(t, *params):
        result = np.zeros_like(t)
        for i in range(0, len(params), 2):
            freq_idx = i // 2
            if freq_idx >= len(top_freqs):
                continue
            A, phi = params[i], params[i+1]
            result += A * np.sin(2*np.pi*top_freqs[freq_idx]*t + phi)
        return result

    initial_guess = [1.0, 0.0] * len(top_freqs)
    
    try:
        params, _ = curve_fit(multi_harmonic_model, t_train, y_train,
                             p0=initial_guess, maxfev=10000)
    except Exception as e:
        print(f"Ошибка подгонки модели для p={p:.1f}: {str(e)}")
        continue
    
    # Прогнозирование
    try:
        S_full = multi_harmonic_model(t_full, *params)
        residual_train = y_train - multi_harmonic_model(t_train, *params)
        trend_smoothed = lowess(residual_train, t_train, frac=0.2, it=3)[:, 1]
        trend_coef = np.polyfit(t_train, trend_smoothed, 1)
        trend_full = np.polyval(trend_coef, t_full)
        y_pred = S_full + trend_full
    except Exception as e:
        print(f"Ошибка прогнозирования для p={p:.1f}: {str(e)}")
        continue

    # Полная визуализация
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2)
    
    # Прогноз
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_train, y_train, 'ko', markersize=3, alpha=0.7, label='Обучающие данные')
    ax1.plot(t_test, y_test, 'ro', markersize=4, alpha=0.7, label='Тестовые данные')
    ax1.plot(t_test, y_pred[test_mask], 'b--', linewidth=1.5, label='Прогноз')
    ax1.axvline(t_full[split_idx], color='g', linestyle='--', label='Граница train/test')
    ax1.set_title(f'Прогноз временного ряда (пропущено {p*100:.1f}% данных)')
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Значение')
    ax1.legend()
    ax1.grid(True)
    
    # Тренд
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_full, trend_full, 'r-')
    ax2.set_title('Выделенный тренд')
    ax2.grid(True)
    
    # Сезонность
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_full, S_full, 'b-')
    ax3.set_title('Сезонная составляющая')
    ax3.grid(True)
    
    # Периодограмма
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(freq, power, 'g-')
    ax4.set_title('Периодограмма Ломба-Скаргла')
    ax4.set_xlabel('Частота')
    ax4.set_ylabel('Сила сигнала')
    ax4.grid(True)
    
    # Метрики
    ax5 = fig.add_subplot(gs[3, :])
    test_pred = y_pred[test_mask]
    
    # Расчет метрик
    try:
        rmse = np.sqrt(np.mean((test_pred - y_test)**2))
        mse = np.mean((test_pred - y_test)**2)
        mae = np.mean(np.abs(test_pred - y_test))
        mape = (np.mean(np.abs((test_pred - y_test)/y_test)) * 100) if np.all(y_test != 0) else np.nan
        r2 = 1 - np.sum((test_pred - y_test)**2)/np.sum((y_test - np.mean(y_test))**2)
    except Exception as e:
        print(f"Ошибка расчета метрик для p={p:.1f}: {str(e)}")
        plt.close()
        continue
    
    metrics_text = (
        f'Метрики качества:\n'
        f'RMSE = {rmse:.3f}\n'
        f'MSE = {mse:.3f}\n'
        f'MAE = {mae:.3f}\n'
        f'MAPE = {mape:.3f}%\n'
        f'R² = {r2:.3f}'
    )
    
    ax5.text(0.5, 0.5, metrics_text, 
            ha='center', va='center', 
            fontsize=14, family='monospace')
    ax5.axis('off')
    
    # Сохранение и закрытие
    try:
        plt.tight_layout()
        plt.savefig(f'results_lomb/combined_plot_{p:.1f}.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Ошибка сохранения графиков для p={p:.1f}: {str(e)}")
        plt.close()