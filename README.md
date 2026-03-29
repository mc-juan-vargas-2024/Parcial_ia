# 🐶 Clasificación de Razas de Perros con CNN

**Estudiante:** Juan Jose Vargas Correa
**Asignatura:** Inteligencia Artificial — Parcial CNN
**Dataset:** Stanford Dogs · 120 razas · 20.580 imágenes

---

## 🚀 App en Streamlit

> 🔗 **[Pegar link aquí]()**

---

## 📋 Descripción

Clasificador de razas caninas entrenado con el dataset **Stanford Dogs** usando una red neuronal convolucional (CNN) construida desde cero en TensorFlow/Keras. El modelo recibe una imagen de un perro y predice su raza entre 120 posibles clases, devolviendo además un top-5 de predicciones con sus probabilidades.

---

## 🗂️ Estructura del Proyecto

```
📦 proyecto
 ┣ 📓 cuaderno.ipynb       — notebook completo de entrenamiento
 ┣ 🐍 app.py               — interfaz Streamlit
 ┗ 🧠 mejor_modelo.keras   — modelo entrenado guardado por checkpoint
```

---

## ⚙️ Pipeline

### 1. Adquisición del Dataset

Se descarga directamente desde los servidores de Stanford el archivo de imágenes y las anotaciones XML con los *bounding boxes*.

```
http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
```

**Imágenes originales detectadas: 20.580**

---

### 2. Preprocesamiento de Imágenes

Cada imagen se recorta usando el *bounding box* de su anotación XML para aislar al perro del fondo, y se redimensiona a **128×128 px** en RGB.

```python
img = img.crop((xmin, ymin, xmax, ymax))
img = img.resize((128, 128))
```

Se validan las coordenadas antes del recorte para descartar bounding boxes inválidos (área cero o negativa). Las imágenes procesadas se guardan organizadas en subcarpetas por raza.

---

### 3. Carga y División del Dataset

```python
tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    seed=42,
    image_size=(128, 128),
    batch_size=32
)
```

| Split | Imágenes |
|-------|----------|
| Entrenamiento (90%) | 18.522 |
| Validación (10%) | 2.058 |

Los píxeles se normalizan de `[0, 255]` a `[0.0, 1.0]` con una capa `Rescaling(1./255)`.

---

### 4. Data Augmentation

Transformaciones aleatorias aplicadas por batch solo durante el entrenamiento para reducir el sobreajuste:

| Transformación | Parámetro |
|---------------|-----------|
| `RandomFlip` | horizontal |
| `RandomRotation` | ±5% |
| `RandomZoom` | ±5% |
| `RandomContrast` | ±10% |
| `clip_by_value` | rango [0, 1] |

---

### 5. Arquitectura CNN

Red secuencial con **3 bloques convolucionales** y **2 capas densas** para clasificación final.

```
Entrada: 128 × 128 × 3 (RGB)
    │
    ├─ Data Augmentation
    │
    ├─ Conv2D(64)  → LeakyReLU → MaxPooling  →  64 × 64
    ├─ Conv2D(128) → LeakyReLU → MaxPooling  →  32 × 32
    ├─ Conv2D(256) → LeakyReLU → MaxPooling  →  16 × 16
    │
    ├─ Flatten → 65.536
    ├─ Dense(64) + BatchNorm
    ├─ Dense(32) + BatchNorm
    │
    └─ Dense(120) + Softmax  →  probabilidad por raza
```

| Parámetros totales | Entrenables |
|-------------------|-------------|
| 4.571.608 (17.44 MB) | 4.571.416 |

---

### 6. Entrenamiento

- **Épocas:** 60
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Callback:** `ModelCheckpoint` → guarda `mejor_modelo.keras` cuando mejora `val_loss`

**Resultado del mejor epoch (epoch 21):**

| Métrica | Valor |
|---------|-------|
| `val_loss` | 2.9223 |
| `val_accuracy` | **29.74%** |

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![Pillow](https://img.shields.io/badge/Pillow-imaging-green)

---

*Stanford Dogs Dataset · CNN entrenada desde cero · Juan Jose Vargas Correa*
# Parcial_ia
# Parcial_ia
