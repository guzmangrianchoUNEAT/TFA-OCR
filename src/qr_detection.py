import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesa la imagen aplicando escala de grises, desenfoque y umbralización adaptativa.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen desde {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Mostrar la imagen binarizada
    cv2.imshow("Imagen Binarizada", binary)
    cv2.waitKey(0)
    return binary, image

def detect_qr_contour(binary_image):
    """
    Detecta todos los contornos en la imagen y agrupa aquellos que están cerca.
    """
    # Detectar contornos en la imagen binarizada
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara para combinar contornos relacionados
    mask = np.zeros_like(binary_image)

    # Dibujar todos los contornos detectados en la máscara
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Mostrar la máscara con los contornos iniciales
    cv2.imshow("Contornos Iniciales", mask)
    cv2.waitKey(0)

    # Dilatar la máscara para unir contornos cercanos
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Mostrar la máscara dilatada
    cv2.imshow("Contornos Unidos (Dilatados)", dilated)
    cv2.waitKey(0)

    # Detectar contornos agrupados en la máscara dilatada
    merged_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno más grande (el QR Code completo)
    if merged_contours:
        largest_contour = max(merged_contours, key=cv2.contourArea)
        return largest_contour

    return None

def order_points(pts):
    """
    Ordena los puntos del contorno en el orden: superior izquierda, superior derecha,
    inferior derecha, inferior izquierda.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_perspective_transform(image, contour):
    """
    Realiza una transformación de perspectiva para obtener el QR Code completo.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) != 4:  
        raise ValueError("No se pudo identificar un contorno cuadrilátero.")

    pts = approx.reshape(4, 2)
    rect = order_points(pts)

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def extract_qr_code(image_path):
    """
    Detecta, visualiza y recorta el QR Code de una imagen.
    """
    binary_image, original_image = preprocess_image(image_path)
    contour = detect_qr_contour(binary_image)

    if contour is None:
        raise ValueError("No se detectó ningún QR Code.")

    # Dibujar el contorno detectado en la imagen original
    cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("Contorno Detectado", original_image)
    cv2.waitKey(0)

    # Transformar perspectiva
    qr_code = get_perspective_transform(original_image, contour)
    return qr_code

if __name__ == "__main__":
    image_path = "data/input/qr_code.png"  
    try:
        qr_code = extract_qr_code(image_path)
        cv2.imshow("QR Code Extraído", qr_code)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(e)
