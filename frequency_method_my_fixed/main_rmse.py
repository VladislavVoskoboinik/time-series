import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

# Создаем директорию для результатов
if not os.path.exists('results_test'):
    os.makedirs('results_test')

precentage_of_missed_data = np.arange(0, 1.0, 0.1)  # 0%, 10%, 20%, ..., 90%
rmse_values = []

for p_idx, p in enumerate(precentage_of_missed_data):
    print(f"\nОбработка случая с {int(p*100)}% пропусков...")
    np.random.seed(42)
    
    # Генерация базовых данных с адаптивным размером
    n_points = int(1000 / (1 - p))  # Адаптивный размер
    t_full = np.linspace(0, 20, n_points)  # Изменено здесь
    
    y_true = (
        0.1 * t_full +
        3 * np.sin(2*np.pi*0.8*t_full) +
        2 * np.sin(2*np.pi*1.5*t_full) +
        1.5 * np.sin(2*np.pi*0.3*t_full)
    )
    y_obs = y_true + np.random.normal(0, 0.5, len(t_full))
    
    # Фиксированные тестовые данные (последние 20%)
    split_idx = int(len(t_full)*0.8)
    test_mask = np.zeros(len(t_full), dtype=bool)
    test_mask[split_idx:] = True
    y_test = y_obs[test_mask]  # Определение y_test
    
    run_rmses = []
    
    repetition_num=5000
    for run in range(repetition_num):
        print(f"  Прогон {run+1}/{repetition_num}...")
        np.random.seed(42 + p_idx * 100 + run)
        
        # Генерация обучающей маски
        train_mask = np.zeros(len(t_full), dtype=bool)
        n_attempts = 0
        
        while True:
            candidate_mask = np.random.choice([True, False], size=split_idx, p=[1-p, p])
            if candidate_mask.sum() >= 5 or n_attempts >= 100:
                train_mask[:split_idx] = candidate_mask
                break
            n_attempts += 1
        
        t_train = t_full[train_mask]
        y_train = y_obs[train_mask]
        
        if len(t_train) < 5:
            print(f"    Недостаточно данных ({len(t_train)} точек). Пропуск.")
            run_rmses.append(np.nan)
            continue
        
        try:
            # Lomb-Scargle периодограмма
            ls = LombScargle(t_train, y_train)
            freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=10.0)
            
            if len(power) == 0:
                raise ValueError("Нет данных в периодограмме")
                
            peaks, _ = find_peaks(power, prominence=np.std(power), distance=5)
            if len(peaks) >= 3:
                top_freqs = freq[peaks][np.argsort(power[peaks])[-3:]]
            else:
                top_freqs = freq[np.argsort(power)[-3:]]
            
            # Модель
            def multi_harmonic_model(t, *params):
                result = np.zeros_like(t)
                for i in range(0, len(params), 2):
                    freq_idx = i//2
                    if freq_idx >= len(top_freqs):
                        continue
                    A, phi = params[i], params[i+1]
                    result += A * np.sin(2*np.pi*top_freqs[freq_idx]*t + phi)
                return result
            
            initial_guess = [1.0, 0.0] * min(3, len(top_freqs))
            params, _ = curve_fit(multi_harmonic_model, t_train, y_train, 
                                p0=initial_guess, maxfev=10000)
            
            # Прогнозирование
            S_full = multi_harmonic_model(t_full, *params)
            residual_train = y_train - multi_harmonic_model(t_train, *params)
            
            # Тренд
            trend_smoothed = lowess(residual_train, t_train, frac=0.2, it=3)[:, 1]
            trend_coef = np.polyfit(t_train, trend_smoothed, 1)
            trend_full = np.polyval(trend_coef, t_full)
            y_pred = S_full + trend_full
            
            # Расчет RMSE
            test_pred = y_pred[test_mask]
            rmse = np.sqrt(np.mean((test_pred - y_test)**2))  # Теперь y_test определен
            run_rmses.append(rmse)
            
            # Визуализация для первого прогона
            if run == 0:
                fig = plt.figure(figsize=(20, 20))
                gs = fig.add_gridspec(4, 2)
                axs = [fig.add_subplot(gs[i//2, i%2]) if i < 2 else 
                      fig.add_subplot(gs[2, :]) if i == 2 else 
                      fig.add_subplot(gs[3, :]) for i in range(4)]
                
                axs[0].plot(t_train, y_train, 'ko', markersize=3, label='Train')
                axs[0].plot(t_full[test_mask], y_test, 'ro', markersize=4, label='Test')
                axs[0].plot(t_full[test_mask], test_pred, 'b--', linewidth=1.5, label='Prediction')
                axs[0].axvline(t_full[split_idx], color='g', linestyle='--', label='Split')
                axs[0].set_title(f'Прогноз ({int(p*100)}% пропусков, RMSE={rmse:.2f})')
                axs[0].legend()
                
                for ax, data, title in zip(axs[1:], 
                                        [trend_full, S_full, power],
                                        ['Тренд', 'Сезонность', 'Периодограмма']):
                    ax.plot(t_full if title != 'Периодограмма' else freq, data)
                    ax.set_title(title)
                    ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'results_test/plot_{int(p*100):02d}_run{run:02d}.png', dpi=100)
                plt.close()
        
        except Exception as e:
            print(f"    Ошибка: {str(e)}")
            run_rmses.append(np.nan)
            continue
    
    # Усреднение RMSE
    valid_rmses = [x for x in run_rmses if not np.isnan(x)]
    avg_rmse = np.mean(valid_rmses) if valid_rmses else np.nan
    rmse_values.append(avg_rmse)
    print(f"Средний RMSE: {avg_rmse:.2f}")

# Финальный график
plt.figure(figsize=(12, 6))
plt.plot(precentage_of_missed_data*100, rmse_values, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Процент пропущенных данных (%)', fontsize=12)
plt.ylabel('Средний RMSE', fontsize=12)
plt.title('Зависимость ошибки прогноза от количества пропусков', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 101, 10))
plt.xlim(-5, 105)
plt.tight_layout()
plt.savefig('results_test/rmse_vs_missing.png', dpi=120)
plt.close()

print("\nАнализ завершен. Результаты сохранены в папке 'results'")