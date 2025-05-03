import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


np.random.seed(42)
time_points = np.arange(0, 50, 0.1)
original_data = np.sin(time_points) + np.sin(2*np.pi/3 * time_points) + np.random.normal(0, 0.1, len(time_points))
original_data[np.random.choice(len(original_data), 50, replace=False)] = np.nan

# Нормализация данных
data_min = np.nanmin(original_data)
data_max = np.nanmax(original_data)
data_range = data_max - data_min + 1e-8
normalized_data = (original_data - data_min) / data_range

# Создание датасета с сохранением индексов
def create_sequence_dataset(series, window_size):
    X, y, time_indices = [], [], []
    for i in range(len(series) - window_size):
        window = series[i:i+window_size]
        target = series[i+window_size]
        
        if np.isnan(target):
            continue
            
        X.append(np.nan_to_num(window, nan=0.0))
        y.append(target)
        time_indices.append(i + window_size)
    return np.array(X), np.array(y), np.array(time_indices)

window_size = 20
X, y, time_indices = create_sequence_dataset(normalized_data, window_size)

# Разделение данных
train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]
time_train, time_test = time_indices[:train_split], time_indices[train_split:]

# Подготовка данных для модели
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Построение модели
model = tf.keras.Sequential([
    tf.keras.Input(shape=(window_size, 1)),
    layers.Masking(mask_value=0.0),
    layers.LSTM(64, kernel_initializer='glorot_uniform'),
    layers.Dense(1, activation='linear')
])

# Функция потерь
def safe_loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.cond(
        tf.size(y_true) > 0,
        lambda: tf.reduce_mean(tf.square(y_true - y_pred)),
        lambda: tf.constant(0.0)
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=safe_loss
)

# Обучение
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

# Прогнозирование
test_predictions = model.predict(X_test).flatten()
test_predictions_orig = (test_predictions * data_range) + data_min

# Подготовка данных для визуализации
y_test_orig = (y_test * data_range) + data_min
test_times = time_points[time_test]
valid_mask = ~np.isnan(y_test_orig)
valid_times = test_times[valid_mask]
valid_true = y_test_orig[valid_mask]
valid_pred = test_predictions_orig[valid_mask]

# Расчет метрик
if len(valid_true) > 0:
    mse = mean_squared_error(valid_true, valid_pred)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
else:
    print("Нет данных для оценки метрик")

# Визуализация 
plt.figure(figsize=(14, 7))

# Исходные данные - точки
plt.scatter(
    time_points, 
    original_data,
    label='Исходные данные',
    color='blue',
    s=10,
    alpha=0.5,
    zorder=2
)

# Прогнозы
plt.plot(
    valid_times,
    valid_pred,
    label='Прогноз',
    color='orange',
    linewidth=2,
    zorder=3
)

# Пропущенные значения
nan_indices = np.where(np.isnan(original_data))[0]
plt.scatter(
    time_points[nan_indices], 
    [data_min] * len(nan_indices),
    color='red',
    marker='x',
    label='Пропуски',
    s=50,
    zorder=4
)

plt.title('Прогнозирование временного ряда с пропусками', fontsize=14)
plt.xlabel('Временная ось', fontsize=12)
plt.ylabel('Значение', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()