import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from metrics import root_mean_squared_error, calculate_all_metrics

# from metrics import calculate_all_metrics


class HoltWinters:
    """
    Класс реализующий трехпараметрическое экспоненциальное сглаживание Хольта-Винтерса
    для прогнозирования временных рядов с трендом и сезонностью.
    """

    def __init__(self, series, seasonal_periods=0, alpha=0.3, beta=0.1, gamma=0.3):
        """
        Инициализация модели Хольта-Винтерса.

        Параметры:
        series (array-like): Исходный временной ряд
        seasonal_periods (int): Длина сезонного цикла
        alpha (float): Параметр сглаживания уровня (0 < alpha < 1)
        beta (float): Параметр сглаживания тренда (0 < beta < 1)
        gamma (float): Параметр сглаживания сезонности (0 < gamma < 1)
        """
        self.series = np.array(series)
        self.seasonal_periods = seasonal_periods
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.level = None
        self.trend = None
        self.seasonal = None
        self.fitted = None

    def initialize_components(self):
        """
        Инициализация начальных значений компонент: уровня, тренда и сезонности
        """
        # Разделяем ряд на сезоны
        seasons = len(self.series) // self.seasonal_periods
        season_averages = np.array([
            self.series[self.seasonal_periods * i:self.seasonal_periods * (i + 1)].mean()
            for i in range(seasons)
        ])

        # Инициализация уровня и тренда
        self.level = season_averages[0]
        self.trend = (season_averages[-1] - season_averages[0]) / (seasons - 1)

        # Инициализация сезонных коэффициентов
        self.seasonal = np.zeros(self.seasonal_periods)
        for i in range(self.seasonal_periods):
            season_slice = self.series[i::self.seasonal_periods][:seasons]
            self.seasonal[i] = np.mean(season_slice / season_averages)

    def fit(self):
        """
        Обучение модели на исторических данных
        """
        self.initialize_components()

        n = len(self.series)
        self.fitted = np.zeros(n)

        for t in range(n):
            if t == 0:
                self.fitted[t] = self.level
            else:
                # Индекс сезона
                s = t % self.seasonal_periods
                # Предыдущий индекс сезона
                s_prev = (t - self.seasonal_periods) % self.seasonal_periods

                # Прогноз на текущий момент
                y_hat = (self.level + self.trend) * self.seasonal[s_prev]
                self.fitted[t] = y_hat

                # Обновление компонент
                level_prev = self.level
                # Обновление уровня
                self.level = self.alpha * (self.series[t] / self.seasonal[s_prev]) + \
                             (1 - self.alpha) * (level_prev + self.trend)

                # Обновление тренда
                self.trend = self.beta * (self.level - level_prev) + \
                             (1 - self.beta) * self.trend

                # Обновление сезонности
                self.seasonal[s] = self.gamma * (self.series[t] / self.level) + \
                                   (1 - self.gamma) * self.seasonal[s_prev]

                rmse = root_mean_squared_error(self.series, self.fitted)
        return self.fitted, rmse

    def forecast(self, steps):
        """
        Прогнозирование будущих значений

        Параметры:
        steps (int): Количество шагов для прогноза

        Возвращает:
        array: Массив прогнозных значений
        """
        forecast = np.zeros(steps)

        for t in range(steps):
            # Определяем индекс сезона для прогноза
            season_index = (len(self.series) + t) % self.seasonal_periods
            # Рассчитываем прогноз
            forecast[t] = (self.level + (t + 1) * self.trend) * self.seasonal[season_index]

        return forecast
    # def plot_series(self, f)


# Пример использования:

def optimized_coefs(series: np.array, seasonal_periods: int):
    best_rmse, best_alpha, best_beta, best_gamma = 1e9, 0.0, 0.0, 0.0
    for alpha_i in np.arange(0, 1, 0.1):
        for beta_i in np.arange(0, 1, 0.1):
            for gamma_i in np.arange(0, 1, 0.1):
                model = HoltWinters(series, seasonal_periods=12, alpha=alpha_i, beta=beta_i, gamma=gamma_i)
                fitted_data, rmse = model.fit()
                if rmse < best_rmse:
                    best_alpha = alpha_i
                    best_beta = beta_i
                    best_gamma = gamma_i
                    best_rmse = rmse
    return best_alpha, best_beta, best_gamma, best_rmse


def initialize_train(series, seasonal_periods: int, forecast_horizon = 24):
    # Инициализируем и обучаем модель
    best_alpha, best_beta, best_gamma, best_rmse = optimized_coefs(series, seasonal_periods)
    # print({"best_alpha:":best_alpha, "best_beta: ": best_beta, "best_gamma: ":best_gamma, "best_rmse: ": best_rmse})
    model = HoltWinters(series, seasonal_periods, alpha=best_alpha, beta=best_beta, gamma=best_gamma)
    fitted_values, rmse = model.fit()

    # Делаем прогноз на 12 периодов вперед
    forecast_values = model.forecast(forecast_horizon)

    # Создаем временные индексы для графика
    time_index = np.arange(len(series))
    # Начинаем прогноз с последней точки исторических данных для непрерывности
    forecast_index = np.arange(len(series) - 1, len(series) + len(forecast_values))
    # Добавляем последнюю точку fitted values к прогнозу для непрерывности
    forecast_values = np.concatenate(([fitted_values[-1]], forecast_values))

    # Создаем график
    plt.figure(figsize=(12, 6))

    # Настраиваем внешний вид графика
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(time_index, series, 'o-', label='Исходные данные', color='blue', markersize=4)
    plt.plot(time_index, fitted_values, '--', label='Подобранные значения', color='green', linewidth=2)
    plt.plot(forecast_index, forecast_values, '--', label='Прогноз', color='red', linewidth=2)

    print(calculate_all_metrics(series, forecast_values[1::]))
    # Вычисляем метрики только для fitted values

    # Добавляем оформление
    plt.title('Прогнозирование временного ряда методом Хольта-Винтерса', fontsize=14, pad=15)
    plt.xlabel('Время', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.legend(fontsize=10)
    plt.axvline(x=len(series) - 1, color="green").set_linestyle("--")
    plt.text(len(series), 3, "Forecast boarder", ha="left")
    # Добавляем отступы для лучшей читаемости
    plt.tight_layout()

    # Показываем график
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.save(f"holt_winter_result_{current_datetime}.jpg")
    plt.show()
    print(rmse)

    # Выводим результаты
    '''''''''
    print("Исходные значения:")
    print(y)
    print("\nПрогноз на 12 периодов вперед:")
    print(forecast_values)
    '''''''''


