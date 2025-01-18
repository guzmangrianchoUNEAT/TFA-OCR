import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from segment_characters import segment_characters_with_lines

def predict_word_with_spaces(image_path, model_path, label_to_index_path):
    """
    Predice texto con espacios y saltos de línea a partir de una imagen de entrada.
    """
    segmented_dir = "data/segmented_chars"
    char_images = segment_characters_with_lines(image_path, segmented_dir)

    print("Cargando el modelo...")
    model = load_model(model_path)
    print("Cargando el mapeo de etiquetas...")
    label_to_index = np.load(label_to_index_path, allow_pickle=True).item()
    index_to_label = {v: k for k, v in label_to_index.items()}

    substitution_map = {"n1": "ñ", "N1": "Ñ"}

    predicted_text = ""
    for char_path in char_images:
        if char_path == "SPACE":
            predicted_text += " "
        elif char_path == "NEWLINE":
            predicted_text += "\n"
        else:
            image = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE) / 255.0
            image = image.reshape(1, 32, 32, 1)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = index_to_label[predicted_class]
            predicted_text += substitution_map.get(predicted_label, predicted_label)

    print(f"\nTexto predicho:\n{predicted_text}")

    # Exportar el texto predicho a un archivo .txt
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predicted_text.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(predicted_text)

    print(f"Texto exportado a: {output_path}")

    return predicted_text

if __name__ == "__main__":
    input_image = "data/input/hola.png"
    model_path = "models/cnn_model.keras"
    label_to_index_path = "models/label_to_index.npy"
    predict_word_with_spaces(input_image, model_path, label_to_index_path)
