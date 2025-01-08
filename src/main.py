import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, Toplevel, Label, Scrollbar, Text, END
from segment_characters import segment_characters_with_lines
from predict_word import predict_word_with_spaces

def select_image():
    """
    Abre un cuadro de diálogo para seleccionar una imagen desde el explorador de archivos.
    :return: Ruta de la imagen seleccionada.
    """
    root = Tk()
    root.withdraw()  # Ocultar la ventana principal de Tk
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")]
    )
    root.destroy()  # Cerrar el contexto de Tk para evitar que bloquee el programa
    return file_path

def show_text_window(predicted_text):
    """
    Muestra una ventana con el texto predicho y permite continuar el programa al cerrarla.
    :param predicted_text: El texto predicho.
    """
    window = Tk()
    window.title("Texto Predicho")
    window.geometry("600x400")
    window.attributes("-topmost", True) 
    window.update() 


    # Etiqueta de título
    label = Label(window, text="Texto Predicho", font=("Arial", 16, "bold"))
    label.pack(pady=10)

    # Área de texto con scroll
    text_area = Text(window, wrap="word", font=("Arial", 14))
    text_area.insert(END, predicted_text)
    text_area.config(state="disabled")  # Bloquear edición del texto
    text_area.pack(expand=True, fill="both", padx=10, pady=10)

    scrollbar = Scrollbar(text_area, command=text_area.yview)
    text_area.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    # Iniciar el bucle de la ventana
    window.mainloop()

def main():
    # Seleccionar imagen
    print("Seleccione una imagen desde el explorador de archivos...")
    image_path = select_image()
    if not image_path:
        print("No se seleccionó ninguna imagen. Cerrando.")
        return

    print(f"Imagen seleccionada: {image_path}")

    # Segmentar caracteres
    segmented_dir = "data/segmented_chars"
    print("Segmentando caracteres...")
    segment_characters_with_lines(image_path, segmented_dir)

    # Predecir texto
    model_path = "models/cnn_model.keras"
    label_to_index_path = "models/label_to_index.npy"
    print("Prediciendo texto...")
    predicted_text = predict_word_with_spaces(image_path, model_path, label_to_index_path)

    # Mostrar el texto predicho en una ventana
    show_text_window(predicted_text)

    # Continuar en la consola después de cerrar la ventana
    print(f"\nTexto predicho mostrado: {predicted_text}")
    print("Programa finalizado correctamente.")

if __name__ == "__main__":
    main()
