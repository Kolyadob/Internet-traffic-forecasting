import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Загрузка и предобработка данных
df = pd.read_csv('network_traffic.csv')
df.columns = ['No', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
# Парсинг времени и установка индекса
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f').dt.time
date_str = '2025-05-12'
df['Timestamp'] = pd.to_datetime(date_str + ' ' + df['Time'].astype(str))
df.set_index('Timestamp', inplace=True)

# 2. Аггрегация трафика по секундам
traffic = df['Length'].resample('1S').sum().fillna(0)

# 3. Масштабирование
scaler = StandardScaler()
volume = traffic.values.reshape(-1, 1)
volume_scaled = scaler.fit_transform(volume)

# 4. Формирование выборок с автокорреляцией
timesteps = 10
X_seq, X_auto, y = [], [], []
for i in range(len(volume_scaled) - timesteps):
    seq = volume_scaled[i:i+timesteps].flatten()
    auto_corr = pd.Series(seq).autocorr(lag=1)
    X_seq.append(seq)
    X_auto.append(auto_corr)
    y.append(volume_scaled[i+timesteps][0])
X_seq = np.array(X_seq).reshape(-1, timesteps, 1)
X_auto = np.array(X_auto).reshape(-1, 1)
y = np.array(y)
split = int(0.8 * len(X_seq))
X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
X_auto_train, X_auto_test = X_auto[:split], X_auto[split:]
y_train, y_test = y[:split], y[split:]

# 5. Функция построения LSTM
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

es = EarlyStopping(patience=50, restore_best_weights=True)

# 6. Обучение базовой LSTM с сохранением истории
model_lstm = build_lstm_model((timesteps, 1))
history_lstm = model_lstm.fit(
    X_seq_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 7. Обучение гибридной LSTM-DNN с историей
input_seq = Input(shape=(timesteps, 1))
x_seq = LSTM(64)(input_seq)
input_auto = Input(shape=(1,))
x_auto = Dense(16, activation='relu')(input_auto)
x = Concatenate()([x_seq, x_auto])
output = Dense(1)(x)
model_hybrid = Model(inputs=[input_seq, input_auto], outputs=output)
model_hybrid.compile(optimizer='adam', loss='mse')
history_hybrid = model_hybrid.fit(
    [X_seq_train, X_auto_train], y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 8. Прогноз и метрики
y_pred_lstm = model_lstm.predict(X_seq_test)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

y_pred_hybrid = model_hybrid.predict([X_seq_test, X_auto_test])
mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)
rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
print(f"LSTM MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}")
print(f"LSTM-DNN MAE: {mae_hybrid:.4f}, RMSE: {rmse_hybrid:.4f}")

# 9. Визуализация
# График потерь обучения
plt.figure()
plt.plot(history_lstm.history['loss'], label='train loss LSTM')
plt.plot(history_lstm.history['val_loss'], label='val loss LSTM')
plt.title('LSTM Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history_hybrid.history['loss'], label='train loss Hybrid')
plt.plot(history_hybrid.history['val_loss'], label='val loss Hybrid')
plt.title('Hybrid LSTM-DNN Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# График фактического vs предсказанного трафика
plt.figure()
plt.plot(y_test, label='Actual Traffic')
plt.plot(y_pred_lstm, label='Predicted LSTM')
plt.plot(y_pred_hybrid, label='Predicted Hybrid')
plt.title('Actual vs Predicted Traffic Volume')
plt.xlabel('Time Step')
plt.ylabel('Scaled Traffic Volume')
plt.legend()
plt.show()
