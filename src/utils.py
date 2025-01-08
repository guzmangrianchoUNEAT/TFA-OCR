from PIL import Image, ImageDraw, ImageFont
import os

def generate_image(text, font_path, output_dir, size=(200, 200), font_size=100):
    """
    Genera una imagen con el texto dado y la guarda en el directorio especificado.
    
    :param text: Texto a renderizar (una sola letra o número).
    :param font_path: Ruta al archivo de fuente (.ttf).
    :param output_dir: Directorio donde se guardarán las imágenes generadas.
    :param size: Tamaño de la imagen (ancho, alto).
    :param font_size: Tamaño de la fuente.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{text}.png")

    # Crear imagen en blanco
    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error cargando la fuente: {e}")
        return

    # Calcular posición para centrar el texto
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]  # Cambiado a textbbox
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Dibujar el texto
    draw.text(position, text, fill="black", font=font)

    # Guardar la imagen
    image.save(output_path)
    print(f"Imagen generada: {output_path}")

if __name__ == "__main__":
    # Configuración
    output_dirs = {
        "mayus": "data/examples/mayus",
        "minus": "data/examples/minus",
        "nums": "data/examples/nums"
    }
    font_path = "assets/KidsOnly.otf"  # Cambia esto a la ruta de la fuente en tu sistema

    # Generar letras mayúsculas
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        generate_image(letter, font_path, output_dirs["mayus"])

    # Generar letras minúsculas
    for letter in "abcdefghijklmnopqrstuvwxyz":
        generate_image(letter, font_path, output_dirs["minus"])

    # Generar números
    for number in "0123456789":
        generate_image(number, font_path, output_dirs["nums"])