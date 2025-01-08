import cv2
import os
import numpy as np

def resize_with_relative_padding(image, max_dimension, target_size=(32, 32)):
    """
    Redimensiona la imagen manteniendo el tamaño relativo al carácter más grande,
    centrando en un fondo negro.
    """
    h, w = image.shape
    target_h, target_w = target_size

    if h == 0 or w == 0 or max_dimension == 0:
        raise ValueError(f"Dimensiones inválidas detectadas: h={h}, w={w}, max_dimension={max_dimension}")

    scale = 0.9 * min(target_w / max_dimension, target_h / max_dimension)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    padded_image = np.zeros(target_size, dtype=np.uint8)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

def invert_colors(image_path):
    """
    Invierte los colores de una imagen (de negro a blanco y viceversa).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inverted_image = cv2.bitwise_not(image)
    cv2.imwrite(image_path, inverted_image)

def merge_vertical_contours(contours, vertical_threshold):
    """
    Fusiona contornos cercanos verticalmente dentro de un umbral dado.
    """
    merged_contours = []
    used = [False] * len(contours)

    for i, contour_a in enumerate(contours):
        if used[i]:
            continue

        x_a, y_a, w_a, h_a = cv2.boundingRect(contour_a)
        merged = contour_a

        for j, contour_b in enumerate(contours):
            if i == j or used[j]:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(contour_b)

            # Verificar si los contornos están alineados verticalmente y cercanos
            if abs(x_a - x_b) < min(w_a, w_b) and abs(y_a + h_a - y_b) <= vertical_threshold:
                merged = np.vstack((merged, contour_b))
                used[j] = True

        merged_contours.append(merged)
        used[i] = True

    return merged_contours

def segment_lines(image):
    """
    Segmenta las líneas de una imagen binarizada.
    """
    horizontal_projection = np.sum(image, axis=1)
    line_indices = np.where(horizontal_projection > 0)[0]

    lines = []
    start_idx = line_indices[0]
    for i in range(1, len(line_indices)):
        if line_indices[i] != line_indices[i - 1] + 1:
            lines.append((start_idx, line_indices[i - 1]))
            start_idx = line_indices[i]
    lines.append((start_idx, line_indices[-1]))

    return [(image[start:end, :], start, end) for start, end in lines]

def segment_characters_with_lines(image_path, output_dir, space_threshold=50, vertical_threshold=180):
    """
    Segmenta las líneas y caracteres de una imagen, detecta espacios y guarda los recortes.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"No se pudo cargar una imagen válida desde {image_path}")

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    lines = segment_lines(binary)

    char_images = []
    line_counter = 0

    for line_image, _, _ in lines:
        contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fusionar contornos verticalmente
        contours = merge_vertical_contours(contours, vertical_threshold)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

        max_dimension = max((max(cv2.boundingRect(c)[2:]) for c in contours if cv2.contourArea(c) > 50), default=1)

        prev_x = None
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            if w * h > 50:
                if prev_x is not None and (x - prev_x) > space_threshold:
                    char_images.append("SPACE")

                char = line_image[y:y + h, x:x + w]
                resized_char = resize_with_relative_padding(char, max_dimension, (32, 32))
                char_path = os.path.join(output_dir, f"line_{line_counter}_char_{len(char_images)}.png")
                cv2.imwrite(char_path, resized_char)
                char_images.append(char_path)

                prev_x = x + w

        char_images.append("NEWLINE")  # Agregar indicador de nueva línea
        line_counter += 1

    for char_path in char_images:
        if char_path not in {"SPACE", "NEWLINE"}:
            invert_colors(char_path)

    print(f"Caracteres segmentados guardados en: {output_dir}")
    return char_images

if __name__ == "__main__":
    input_image = "data/input/parrafo3.png"
    output_dir = "data/segmented_chars"
    segment_characters_with_lines(input_image, output_dir)
