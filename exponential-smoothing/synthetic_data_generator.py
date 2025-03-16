import numpy as np
import pandas as pd


def generate_complex_series(size: int) -> pd.Series:
    """
    Генерирует временной ряд, содержащий:
    - Линейный тренд
    - Годовую сезонность (период 12)
    - Квартальную сезонность (период 4)
    - Случайный шум
    
    Параметры:
    ----------
    size : int
        Длина временного ряда
        
    Возвращает:
    -----------
    pd.Series
        Временной ряд
    """
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


if __name__ == "__main__":
    # Пример использования
    series = generate_complex_series(60)
    
    # Визуализация
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Временной ряд')
    plt.title('Синтетический временной ряд с трендом, сезонностью и шумом')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.show() 