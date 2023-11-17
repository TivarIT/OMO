import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
housing = fetch_california_housing()
data = housing.data
target = housing.target

# Нормализация данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

# Создание модели линейной регрессии с использованием TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Входной слой
    tf.keras.layers.Dense(1)  # Выходной слой с одним нейроном (линейная регрессия)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=100, verbose=1)

# Вывод коэффициентов модели
weights, biases = model.layers[1].get_weights()
print("Коэффициенты модели:")
print("Веса:", weights)
print("Смещение:", biases)