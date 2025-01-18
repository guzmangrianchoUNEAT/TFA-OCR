from pyzbar.pyzbar import decode
import os
import cv2
from qr_detection import extract_qr_code

def decode_qr_code(qr_image):
    decoded_objects = decode(qr_image)
    if not decoded_objects:
        return "No se pudo decodificar el QR Code."
    
    decoded_data = decoded_objects[0].data.decode("utf-8")
    # Guardar el QR Code decodificado en un archivo de texto
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qr_code_data.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded_data)

    print(f"Datos del QR Code exportados a: {output_path}")
    
    return decoded_data

def main(image_path):
    try:
        qr_code = extract_qr_code(image_path)
        decoded_data = decode_qr_code(qr_code)
        print(f"Datos del QR Code: {decoded_data}")

        cv2.imshow("QR Code Detectado", qr_code)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    image_path = "data/input/qr_code_persp.png"
    main(image_path)
