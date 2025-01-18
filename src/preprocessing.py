import os
import cv2
import numpy as np

def process_and_unify_data(raw_dir, unified_dir, size=(32, 32), augmentations_per_image=5):
    """
    Procesa, aplica rotaciones y desplazamientos, y unifica los datos de las carpetas raw.
    :param raw_dir: Directorio con las carpetas de cada estudiante y subcarpetas (Mayusculas, Minusculas, Numeros).
    :param unified_dir: Directorio destino donde se almacenarán las imágenes procesadas.
    :param size: Tamaño al que se redimensionarán las imágenes (ancho, alto).
    :param augmentations_per_image: Número de variaciones generadas por cada imagen original.
    """
    # Crear subcarpetas para Mayusculas, Minusculas y Numeros
    mayus_dir = os.path.join(unified_dir, 'mayus')
    minus_dir = os.path.join(unified_dir, 'minus')
    nums_dir = os.path.join(unified_dir, 'nums')

    os.makedirs(mayus_dir, exist_ok=True)
    os.makedirs(minus_dir, exist_ok=True)
    os.makedirs(nums_dir, exist_ok=True)

    def augment_image(image):
        """Aplica rotaciones y desplazamientos aleatorios a una imagen."""
        augmented_images = []
        for _ in range(augmentations_per_image):
            # Rotación aleatoria
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            
            # Desplazamiento aleatorio
            tx = np.random.uniform(-5, 5)  
            ty = np.random.uniform(-5, 5) 
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            shifted = cv2.warpAffine(rotated, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            
            augmented_images.append(shifted)
        return augmented_images

    # Recorrer cada carpeta de estudiante
    for student in os.listdir(raw_dir):
        student_path = os.path.join(raw_dir, student)
        if os.path.isdir(student_path):
            # Recorrer categorías (Mayusculas, Minusculas, Numeros)
            for category, target_dir in zip(['Mayusculas', 'Minusculas', 'Numeros'], [mayus_dir, minus_dir, nums_dir]):
                category_path = os.path.join(student_path, category)
                if os.path.isdir(category_path):
                    # Procesar cada imagen en la categoría
                    for image_name in os.listdir(category_path):
                        # Leer el carácter del nombre del archivo (sin extensión)
                        char = os.path.splitext(image_name)[0]

                        # Crear subcarpeta para el carácter dentro de la categoría
                        char_dir = os.path.join(target_dir, char)
                        os.makedirs(char_dir, exist_ok=True)

                        # Leer y preprocesar la imagen
                        image_path = os.path.join(category_path, image_name)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        resized = cv2.resize(image, size)
                        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

                        # Generar variaciones
                        variations = augment_image(binary)
                        
                        # Guardar la imagen original
                        original_name = f"{student}_{category}_{image_name}"
                        original_path = os.path.join(char_dir, original_name)
                        cv2.imwrite(original_path, binary)

                        # Guardar las variaciones
                        for idx, variation in enumerate(variations):
                            variation_name = f"{student}_{category}_{char}_aug{idx}.png"
                            variation_path = os.path.join(char_dir, variation_name)
                            cv2.imwrite(variation_path, variation)

    print(f"Datos unificados y preprocesados con rotaciones y desplazamientos en: {unified_dir}")

# Ejemplo de uso
if __name__ == "__main__":
    raw_dir = 'data/raw' 
    unified_dir = 'data/unified' 
    process_and_unify_data(raw_dir, unified_dir)
