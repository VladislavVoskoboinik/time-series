import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Вычисляет среднеквадратичную ошибку (Mean Squared Error, MSE).

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    float: Значение MSE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Вычисляет корень из среднеквадратичной ошибки (Root Mean Squared Error, RMSE).

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    float: Значение RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Вычисляет среднюю абсолютную ошибку (Mean Absolute Error, MAE).

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    float: Значение MAE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Вычисляет взвешенную абсолютную процентную ошибку (Weighted Absolute Percentage Error, WAPE).

    WAPE = Σ|y_true - y_pred| / Σ|y_true| * 100%

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    float: Значение WAPE в процентах
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def r_squared(y_true, y_pred):
    """
    Вычисляет коэффициент детерминации (R²).
    R² показывает долю дисперсии зависимой переменной, объясняемую моделью.

    R² = 1 - SS_res/SS_tot
    где:
    SS_res - сумма квадратов остатков
    SS_tot - общая сумма квадратов

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    float: Значение R² (от 0 до 1, где 1 означает идеальную подгонку)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Среднее значение истинных данных
    y_mean = np.mean(y_true)

    # Общая сумма квадратов (total sum of squares)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Сумма квадратов остатков (residual sum of squares)
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Вычисляем R²
    r2 = 1 - (ss_res / ss_tot)

    return r2


def calculate_all_metrics(y_true, y_pred):
    """
    Вычисляет все метрики качества прогноза.

    Параметры:
    y_true (array-like): Истинные значения
    y_pred (array-like): Предсказанные значения

    Возвращает:
    dict: Словарь с значениями всех метрик
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'WAPE': weighted_absolute_percentage_error(y_true, y_pred),
        'R2': r_squared(y_true, y_pred)
    }
    return metrics