import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from holt_winter_model import initialize_train


def generate_series_1(size: int) -> pd.Series:
    t = np.arange(size)
    series = 2 + 0.1 * t + np.sin(2 * np.pi * t / 12)
    return pd.Series(series)


def generate_series_2(size: int) -> pd.Series:
    t = np.arange(size)
    series = np.exp(0.01 * t) + np.cos(2 * np.pi * t / 12)
    return pd.Series(series)


def generate_series_3(size: int) -> pd.Series:
    t = np.arange(size)
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size)
    series = 2 + 0.1 * t + np.sin(2 * np.pi * t / 12) + noise
    return pd.Series(series)


def generate_series_4(size: int) -> pd.Series:
    t = np.arange(size)
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size)
    series = np.exp(0.01 * t) + np.cos(2 * np.pi * t / 12) + noise
    return pd.Series(series)


def generate_series_5(size: int) -> pd.Series:
    t = np.arange(size)
    series = 2 + 0.1 * t + np.sin(2 * np.pi * t / 12) + np.cos(2 * np.pi * t / 6)
    return pd.Series(series)

def generate_series_6(size: int) -> pd.Series:
    t = np.arange(size)
    # Линейный тренд
    trend = 0.15 * t
    # Годовая сезонность
    yearly_seasonality = 2 * np.sin(2 * np.pi * t / 12)
    # Квартальная сезонность
    quarterly_seasonality = 1.5 * np.cos(2 * np.pi * t / 4)
    # Случайный шум
    np.random.seed(42)  # для воспроизводимости
    noise = np.random.normal(0, 0.5, size)
    # Объединяем все компоненты
    series = trend + yearly_seasonality + quarterly_seasonality + noise

    return pd.Series(series)


'''''
# Генерация и визуализация временных рядов
n = 120  # Количество периодов

series1 = generate_series_1(n)
series2 = generate_series_2(n)
series3 = generate_series_3(n)
series4 = generate_series_4(n)
series5 = generate_series_5(n)

# Создание фигуры и подграфиков
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8, 15))

# Построение графиков
axs[0].plot(series1)
axs[0].set_title('Линейный тренд с синусоидальной сезонностью')

axs[1].plot(series2)
axs[1].set_title('Экспоненциальный тренд с косинусоидальной сезонностью')

axs[2].plot(series3)
axs[2].set_title('Линейный тренд с шумом и синусоидальной сезонностью')

axs[3].plot(series4)
axs[3].set_title('Экспоненциальный тренд с шумом и косинусоидальной сезонностью')

axs[4].plot(series5)
axs[4].set_title('Сложный тренд с комбинированной сезонностью')

# Удаление осей между графиками
for ax in axs:
    ax.set_xlabel('Период')
    ax.set_ylabel('Значение')

# Автоматическое расположение элементов
plt.tight_layout()

plt.show()
'''''
if __name__ == "__main__":
    initialize_train(generate_series_1(100), seasonal_periods=12, forecast_horizon=20)
    initialize_train(generate_series_2(100), seasonal_periods=12, forecast_horizon=20)
    initialize_train(generate_series_3(100), seasonal_periods=12, forecast_horizon=20)
    initialize_train(generate_series_4(100), seasonal_periods=12, forecast_horizon=20)
    initialize_train(generate_series_5(100), seasonal_periods=12, forecast_horizon=20)
    initialize_train(generate_series_6(100), seasonal_periods=12, forecast_horizon=20)
