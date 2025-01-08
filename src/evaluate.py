import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def load_examples(example_dir, size=(32, 32)):
    """
    Carga imágenes desde el directorio de ejemplos para realizar predicciones.
    """
    X, y_true, filenames = [], [], []
    for file_name in os.listdir(example_dir):
        file_path = os.path.join(example_dir, file_name)
        if file_name.endswith('.png') and os.path.isfile(file_path):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized = cv2.resize(image, size) / 255.0
                X.append(resized.reshape(32, 32, 1))
                filenames.append(file_name)
                # Extraer etiqueta real del nombre del archivo
                label = file_name.split('.')[0].split('-')[0]
                y_true.append(label)
    return np.array(X), np.array(y_true), filenames

def predict_and_evaluate(model, X, y_true, filenames, label_to_index):
    """
    Realiza predicciones y calcula métricas.
    """
    # Convertir etiquetas reales a índices
    y_true_indices = np.array([label_to_index.get(label, -1) for label in y_true])
    invalid_indices = y_true_indices == -1

    # Filtrar datos inválidos
    if np.any(invalid_indices):
        print(f"Advertencia: Etiquetas no encontradas en el diccionario: {set(y_true[invalid_indices])}")
    valid_indices = ~invalid_indices
    X = X[valid_indices]
    y_true_indices = y_true_indices[valid_indices]
    filenames = np.array(filenames)[valid_indices]

    # Realizar predicciones
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)

    # Invertir el diccionario
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Mostrar predicciones por archivo
    print("\nPredicciones:")
    for filename, pred_class, true_class in zip(filenames, predicted_classes, y_true_indices):
        pred_label = index_to_label.get(pred_class, "UNKNOWN")
        true_label = index_to_label.get(true_class, "UNKNOWN")
        print(f"{filename}: Predicho: {pred_label} | Real: {true_label}")

    # Identificar clases presentes
    present_classes = np.unique(np.concatenate([y_true_indices, predicted_classes]))

    # Calcular métricas finales
    accuracy = accuracy_score(y_true_indices, predicted_classes)
    print(f"\nPorcentaje de aciertos: {accuracy * 100:.2f}%")
    print("\nReporte de clasificación:")
    print(classification_report(
        y_true_indices,
        predicted_classes,
        labels=present_classes,
        target_names=[index_to_label[c] for c in present_classes],
        zero_division=0  # Evitar errores en clases sin predicciones
    ))

    return predicted_classes

def visualize_predictions(X, filenames, predicted_classes, label_to_index):
    """
    Muestra imágenes de ejemplos con sus predicciones.
    """
    index_to_label = {v: k for k, v in label_to_index.items()}
    plt.figure(figsize=(15, 10))
    for i in range(min(len(X), 12)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X[i].reshape(32, 32), cmap='gray')
        plt.title(f"Pred: {index_to_label.get(predicted_classes[i], 'UNKNOWN')}\nFile: {filenames[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Rutas
    example_dir = 'data/examples'
    model_path = 'models/cnn_model.keras'
    label_to_index_path = 'models/label_to_index.npy'

    # Cargar el modelo
    print("Cargando el modelo...")
    model = load_model(model_path)

    # Cargar el diccionario de etiquetas
    print("Cargando el mapeo de etiquetas...")
    label_to_index = np.load(label_to_index_path, allow_pickle=True).item()

    # Cargar imágenes de ejemplo
    print("Cargando imágenes de ejemplo...")
    X, y_true, filenames = load_examples(example_dir)

    # Realizar predicciones y calcular métricas
    print("Realizando predicciones...")
    predicted_classes = predict_and_evaluate(model, X, y_true, filenames, label_to_index)

    # Visualizar ejemplos con predicciones
    print("\nVisualizando ejemplos con predicciones:")
    visualize_predictions(X, filenames, predicted_classes, label_to_index)
