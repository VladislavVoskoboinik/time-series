import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

precentage_of_missed_data = np.linspace(0., 0.9, 10)
for p in precentage_of_missed_data:
    # Генерация данных
    np.random.seed(42)
    t_full = np.linspace(0, 20, 500)
    """
    y_true = (
        0.5 * t_full +
        3 * np.sin(2*np.pi*0.8*t_full) + 
        2 * np.sin(2*np.pi*1.5*t_full) + 
        1.5 * np.sin(2*np.pi*0.3*t_full)
    )
    """
    y_true = (
        0.5 * t_full 
        + 3.0 * np.sin(2*np.pi*0.8*t_full)
        + 2.5 * np.sin(2*np.pi*1.5*t_full)
        + 2.0 * np.sin(2*np.pi*0.3*t_full)
        + 1.8 * np.sin(2*np.pi*2.1*t_full)
        + 1.5 * np.sin(2*np.pi*1.2*t_full)
        + 1.3 * np.sin(2*np.pi*0.7*t_full)
        + 1.1 * np.sin(2*np.pi*2.8*t_full)
        + 0.9 * np.sin(2*np.pi*0.5*t_full)
        + 0.7 * np.sin(2*np.pi*1.8*t_full)
        + 0.5 * np.sin(2*np.pi*3.2*t_full)
        + 0.3 * np.sin(2*np.pi*0.9*t_full)
    )
    y_obs = y_true + np.random.normal(0, 0.5, len(t_full))
    
    # Разделение данных
    split_idx = int(len(t_full)*0.8)
    train_mask = np.full(len(t_full), False)
    train_mask[:split_idx] = np.random.choice([True, False], split_idx, p=[1-p, p])
    test_mask = np.full(len(t_full), False)
    test_mask[split_idx:] = True

    t_train = t_full[train_mask]
    y_train = y_obs[train_mask]
    t_test = t_full[test_mask]
    y_test = y_obs[test_mask]

    # Проверка наличия данных
    if len(t_train) < 2:
        print(f"Пропуск p={p:.1f} - недостаточно данных для анализа")
        continue

    # Lomb-Scargle периодограмма
    try:
        ls = LombScargle(t_train, y_train)
        freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=10.0)
        peaks, _ = find_peaks(power, prominence=np.std(power), distance=5)
        top_freqs = freq[peaks][np.argsort(power[peaks])[-5:]]
    except Exception as e:
        print(f"Ошибка в периодограмме при p={p:.1f}: {str(e)}")
        continue

    # Обучение модели
    def multi_harmonic_model(t, *params):
        result = np.zeros_like(t)
        for i in range(0, len(params), 2):
            A, phi = params[i], params[i+1]
            result += A * np.sin(2*np.pi*top_freqs[i//2]*t + phi)
        return result

    initial_guess = [1.0, 0.0] * len(top_freqs)
    try:
        params, _ = curve_fit(multi_harmonic_model, t_train, y_train, p0=initial_guess, maxfev=10000)
    except Exception as e:
        print(f"Ошибка оптимизации при p={p:.1f}: {str(e)}")
        continue

    # Выделение тренда с помощью временного скользящего среднего
    residual_train = y_train - multi_harmonic_model(t_train, *params)
    
    # Расчет параметров окна (20% от длительности ряда)
    duration = t_train.max() - t_train.min()
    window_size = 0.1 * duration if duration > 0 else 0
    
    trend_smoothed = np.zeros_like(t_train)
    if window_size > 0:
        for i, t in enumerate(t_train):
            # Выбор точек в пределах временного окна
            mask = (t_train >= t - window_size/2) & (t_train <= t + window_size/2)
            if mask.sum() > 0:
                trend_smoothed[i] = residual_train[mask].mean()
    
    # Аппроксимация линейного тренда
    trend_coef = np.polyfit(t_train, trend_smoothed, 1)
    trend_full = np.polyval(trend_coef, t_full)
    S_full = multi_harmonic_model(t_full, *params)
    y_pred = S_full + trend_full

    # Визуализация
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    ax5 = fig.add_subplot(gs[3, :])

    # График прогноза
    ax1.plot(t_train, y_train, 'ko', markersize=3, alpha=0.7, label='Обучающие данные')
    ax1.plot(t_test, y_test, 'ro', markersize=4, alpha=0.7, label='Тестовые данные')
    ax1.plot(t_test, y_pred[test_mask], 'b--', linewidth=1.5, label='Прогноз')
    ax1.axvline(t_full[split_idx], color='g', linestyle='--', label='Граница train/test')
    ax1.set_title(f'Прогноз временного ряда (пропущено {p*100:.1f}% данных)')
    ax1.legend()
    ax1.grid(True)

    # Тренд
    ax2.plot(t_full, trend_full, 'r-')
    ax2.set_title('Выделенный тренд')
    ax2.grid(True)

    # Сезонность
    ax3.plot(t_full, S_full, 'b-')
    ax3.set_title('Сезонная составляющая')
    ax3.grid(True)

    # Периодограмма
    ax4.plot(freq, power, 'g-')
    ax4.set_title('Периодограмма Ломба-Скаргла')
    ax4.grid(True)

    # Метрики
    test_pred = y_pred[test_mask]
    rmse = np.sqrt(np.mean((test_pred - y_test)**2))
    mse = np.mean((test_pred - y_test)**2)
    mae = np.mean(np.abs(test_pred - y_test))
    mape = (np.mean(np.abs((test_pred - y_test) / y_test)) * 100) if np.all(y_test != 0) else np.nan
    r2 = 1 - np.sum((test_pred - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)

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

    plt.tight_layout()
    plt.savefig(f'results/combined_plot_{p:.1f}_ma_direct_more_freq.png', dpi=150, bbox_inches='tight')
    plt.close()