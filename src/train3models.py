import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Cargar los datos
def load_data(base_dir):
    """Carga imágenes y etiquetas desde la estructura de carpetas."""
    X, y = [], []
    for category in ['mayus', 'minus', 'nums']:
        category_dir = os.path.join(base_dir, category)
        if os.path.isdir(category_dir):
            for label in os.listdir(category_dir):
                label_dir = os.path.join(category_dir, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        img_path = os.path.join(label_dir, img_name)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            X.append(cv2.resize(image, (32, 32)))
                            y.append(label) 
    return np.array(X), np.array(y)

# 2. Preprocesar datos
def preprocess_data(X, y):
    """Normaliza imágenes y convierte etiquetas a categóricas."""
    X = X.reshape(-1, 32, 32, 1) / 255.0  
    unique_labels = sorted(set(y))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_index[label] for label in y])
    y = to_categorical(y, num_classes=len(unique_labels))
    return X, y, label_to_index

# 3. Definir múltiples arquitecturas de modelos
def build_model_1(input_shape, num_classes):
    """Modelo CNN básico."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_2(input_shape, num_classes):
    """Modelo CNN con más filtros y capas densas."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_3(input_shape, num_classes):
    """Modelo CNN con capas adicionales y dropout reducido."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Lista de arquitecturas de modelos
models_to_try = [build_model_1, build_model_2, build_model_3]

# 4. Entrenamiento
def train_model(X_train, y_train, X_val, y_val, model_func, label_to_index):
    """Entrena el modelo con aumentación de datos y early stopping."""
    model = model_func(input_shape=(32, 32, 1), num_classes=len(label_to_index))

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,  
        height_shift_range=0.1 
    )
    datagen.fit(X_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[early_stopping],
        verbose=1
    )

    y_val_pred = model.predict(X_val).argmax(axis=1)
    y_val_true = y_val.argmax(axis=1)
    print(classification_report(y_val_true, y_val_pred, target_names=list(label_to_index.keys())))

    return model, history

# 5. Graficar resultados
def plot_metrics(history):
    """Genera gráficos de pérdida y precisión."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.legend()
    plt.show()

# 6. Main
if __name__ == "__main__":
    base_dir = 'data/unifiedFonts' 
    X, y = load_data(base_dir)
    X, y, label_to_index = preprocess_data(X, y)

    os.makedirs('models', exist_ok=True)
    np.save('models/label_to_index.npy', label_to_index)

    print(f"Etiquetas disponibles: {label_to_index}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))

    for i, model_func in enumerate(models_to_try):
        print(f"\nEntrenando modelo {i + 1}...")
        model, history = train_model(X_train, y_train, X_val, y_val, model_func, label_to_index)
        plot_metrics(history)
        model.save(f'models/cnn_model_{i + 1}_manus.keras')
        print(f"Modelo {i + 1} guardado en 'models/cnn_model_{i + 1}_fonts.keras'")
