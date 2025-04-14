import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Генерация данных
np.random.seed(42)
t_full = np.linspace(0, 20, 500)
y_true = (
    0.1 * t_full**1.5 + 
    3 * np.sin(2*np.pi*0.8*t_full) + 
    2 * np.sin(2*np.pi*1.5*t_full) + 
    1.5 * np.sin(2*np.pi*0.3*t_full)
)
y_obs = y_true + np.random.normal(0, 0.3, len(t_full))

# Разделение на обучающую (80%) и тестовую (20%) части
split_idx = int(len(t_full)*0.8)
train_mask = np.full(len(t_full), False)
train_mask[:split_idx] = np.random.choice([True, False], split_idx, p=[0.6, 0.4])  # 60% данных в обучающей
test_mask = np.full(len(t_full), False)
test_mask[split_idx:] = True  # Все тестовые точки считаем наблюдаемыми для чистоты оценки

t_train = t_full[train_mask]
y_train = y_obs[train_mask]
t_test = t_full[test_mask]
y_test = y_obs[test_mask]

# Lomb-Scargle периодограмма
ls = LombScargle(t_train, y_train)
freq, power = ls.autopower(minimum_frequency=0.1, maximum_frequency=3.0)
peaks, _ = find_peaks(power, prominence=np.std(power), distance=10)
top_freqs = freq[peaks][np.argsort(power[peaks])[-5:]]  # Выделить 5 частот

# Визуализация периодограммы
plt.figure(figsize=(10, 4))
plt.plot(freq, power)
plt.xlabel('Частота')
plt.ylabel('Сила сигнала')
plt.title('Периодограмма Лэмба')
plt.grid(True)
plt.show()

# Обучение модели
def multi_harmonic_model(t, *params):
    result = np.zeros_like(t)
    for i in range(0, len(params), 2):
        A, phi = params[i], params[i+1]
        result += A * np.sin(2*np.pi*top_freqs[i//2]*t + phi)
    return result

initial_guess = [1.0, 0.0] * len(top_freqs)
params, _ = curve_fit(multi_harmonic_model, t_train, y_train, p0=initial_guess, maxfev=10000)

# Прогнозирование для всего ряда
S_full = multi_harmonic_model(t_full, *params)
residual_train = y_train - multi_harmonic_model(t_train, *params)
trend_smoothed = lowess(residual_train, t_train, frac=0.2, it=3)[:, 1]
trend_coef = np.polyfit(t_train, trend_smoothed, 1)
trend_full = np.polyval(trend_coef, t_full)
y_pred = S_full + trend_full

# Визуализация
fig, axs = plt.subplots(3, figsize=(15, 16))

# Обучающая и тестовая выборка
axs[0].plot(t_train, y_train, 'ko', markersize=3, alpha=0.7, label='Обучающие данные (с пропусками)')
axs[0].plot(t_test, y_test, 'ro', markersize=4, alpha=0.7, label='Тестовые данные (реальные значения)')
axs[0].plot(t_test, y_pred[test_mask], 'b--', linewidth=1.5, label='Прогноз')
axs[0].axvline(t_full[split_idx], color='g', linestyle='--', label='Граница train/test')
axs[0].set_title('Прогноз временного ряда')
axs[0].set_xlabel('Время')
axs[0].set_ylabel('Значение')
axs[0].legend()
axs[0].grid(True)

# Выделенный тренд
axs[1].plot(t_full, trend_full, 'r-', label='Тренд')
axs[1].set_title('Выделенный тренд')
axs[1].set_xlabel('Время')
axs[1].set_ylabel('Значение')
axs[1].legend()
axs[1].grid(True)

# Сезонность
axs[2].plot(t_full, S_full, 'b-', label='Сезонность')
axs[2].set_title('Сезонность')
axs[2].set_xlabel('Время')
axs[2].set_ylabel('Значение')
axs[2].legend()
axs[2].grid(True)



plt.tight_layout()
plt.show()

# Метрики только для наблюдаемых тестовых точек
test_pred = y_pred[test_mask]
rmse = np.sqrt(np.mean((test_pred - y_test)**2))
mse = np.mean((test_pred - y_test)**2)
mae = np.mean(np.abs(test_pred - y_test))
mape = np.mean(np.abs((test_pred - y_test) / y_test)) * 100 if np.all(y_test != 0) else np.nan
r2 = 1 - np.sum((test_pred - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f'RMSE на тестовой выборке: {rmse:.3f}')
print(f'MSE на тестовой выборке: {mse:.3f}')
print(f'MAE на тестовой выборке: {mae:.3f}')
print(f'MAPE на тестовой выборке: {mape:.3f}%')
print(f'R2 на тестовой выборке: {r2:.3f}')
