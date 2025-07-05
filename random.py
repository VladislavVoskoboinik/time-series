import tensorflow as tf
from tensorflow.keras.layers import GRUCell, Dense

class GRUD(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.gru_cell = GRUCell(units)
        self.decay_layer = Dense(1, activation='sigmoid')  # Decay for missing values

    def call(self, inputs):
        x, mask, delta_t = inputs
        h_prev = tf.zeros((x.shape[0], self.gru_cell.units))
        outputs = []
        for t in range(x.shape[1]):
            # Calculate decay
            decay = self.decay_layer(delta_t[:, t])
            x_t = x[:, t] * mask[:, t] + (1 - mask[:, t]) * decay * h_prev
            h_prev, _ = self.gru_cell(x_t, h_prev)
            outputs.append(h_prev)
        return tf.stack(outputs, axis=1)

# Пример использования
model = GRUD(units=64)
model.compile(optimizer='adam', loss='mse')
model.fit([x_train, mask_train, delta_t_train], y_train, epochs=10)