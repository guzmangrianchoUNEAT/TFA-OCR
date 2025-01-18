from PIL import Image, ImageDraw, ImageFont
import os

def generate_image(text, font_path, output_dir, font_name, size=(32, 32), font_size=16):
    """
    Genera una imagen cuadrada con el texto dado centrado y ajustado según el tipo de carácter y contornos reales.
    
    :param text: Texto a renderizar (una sola letra o número).
    :param font_path: Ruta al archivo de fuente (.ttf).
    :param output_dir: Directorio donde se guardarán las imágenes generadas.
    :param font_name: Nombre de la fuente utilizada.
    :param size: Tamaño de la imagen (ancho, alto).
    :param font_size: Tamaño de la fuente base.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ajustar el nombre de la imagen
    adjusted_text = text.replace("Ñ", "N1").replace("ñ", "n1")
    output_path = os.path.join(output_dir, f"{adjusted_text}_{font_name}.png")

    # Crear imagen cuadrada en blanco
    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error cargando la fuente {font_path}: {e}")
        return

    # Calcular el tamaño del texto y su caja delimitadora
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Ajustar tamaño según el tipo de carácter
    if text.isupper() or text.isdigit():
        scale_factor = 2.3  
    else:
        scale_factor = 2.2 

    adjusted_font_size = int(font_size * scale_factor)
    try:
        font = ImageFont.truetype(font_path, adjusted_font_size)
    except Exception as e:
        print(f"Error ajustando el tamaño de la fuente {font_path}: {e}")
        return

    # Recalcular tamaño del texto con la fuente ajustada
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Calcular posición centrada según la caja delimitadora
    x_offset = (size[0] - text_width) // 2 - bbox[0]
    y_offset = (size[1] - text_height) // 2 - bbox[1]

    # Dibujar el texto centrado
    draw.text((x_offset, y_offset), text, fill="black", font=font)

    # Guardar la imagen
    image.save(output_path)
    print(f"Imagen generada: {output_path}")

if __name__ == "__main__":
    # Configuración
    base_output_dir = "data/unifiedFonts"
    font_dir = "assets/training_fonts"
    
    # Crear carpetas base
    categories = {"mayus": "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ",  
                  "minus": "abcdefghijklmnñopqrstuvwxyz",  
                  "nums": "0123456789"}

    # Procesar todas las fuentes en el directorio
    for font_file in os.listdir(font_dir):
        font_path = os.path.join(font_dir, font_file)
        if not font_file.endswith(".ttf"):
            continue

        font_name = os.path.splitext(font_file)[0]
        print(f"Procesando fuente: {font_file}")

        for category, characters in categories.items():
            for char in characters:
                adjusted_char = char.replace("Ñ", "N1").replace("ñ", "n1")
                char_dir = os.path.join(base_output_dir, category, adjusted_char)
                generate_image(char, font_path, char_dir, font_name)
