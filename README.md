# 🎯 Sistema de Recomendación y Clasificación de Imágenes con Redes Neuronales

Este repositorio contiene dos proyectos desarrollados en Python con TensorFlow:

- 📽️ Un sistema de recomendación de películas usando redes neuronales densas.
- 🖼️ Una red convolucional para clasificar imágenes del juego piedra, papel o tijera.

---

## 🧠 Modelos Desarrollados

### 🔷 Recomendación de Películas (MovieLens 100K)

- **Dataset**: `u.data`
- **Modelo**: Red neuronal densa (MLP)
- **Archivo principal**: `movie_rating_model.py`
- **Métricas de evaluación**:
  - MAE: ~0.74
  - RMSE: ~0.94

### 🔷 Clasificación de Imágenes con CNN

- **Dataset**: `rock_paper_scissors` (de TensorFlow Datasets)
- **Modelo**: Red convolucional con `Conv2D`, `MaxPooling2D`, `Dense`, `Dropout`
- **Archivo principal**: `cnn_classifier.py`
- **Métricas de evaluación**:
  - Accuracy: ~78.2%
  - F1 Score (macro): ~0.775
