import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Cargar el dataset
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("u.data", sep="\t", names=column_names)

# 2. Normalizar IDs (comienzan desde 0)
df["user_id"] = df["user_id"] - 1
df["item_id"] = df["item_id"] - 1

num_users = df["user_id"].nunique()
num_items = df["item_id"].nunique()

# 3. Dividir en train/test
X = df[["user_id", "item_id"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Definir el modelo de recomendación
class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.concat = tf.keras.layers.Concatenate()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_input, item_input = inputs[:, 0], inputs[:, 1]
        user_vec = self.user_embedding(user_input)
        item_vec = self.item_embedding(item_input)
        x = self.concat([user_vec, item_vec])
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# 5. Instanciar y compilar
model = RecommenderModel(num_users, num_items)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Entrenar
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# 7. Evaluar
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")