import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, Toplevel, Label, Scrollbar, Text, Button, END
from segment_characters import segment_characters_with_lines
from predict_word import predict_word_with_spaces
from qr_decode import extract_qr_code, decode_qr_code

def select_image():
    """
    Abre un cuadro de diálogo para seleccionar una imagen desde el explorador de archivos.
    :return: Ruta de la imagen seleccionada.
    """
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")]
    )
    root.destroy()  
    return file_path

def choose_processing_option():
    """
    Abre una ventana para elegir entre procesar texto manuscrito, texto de ordenador o código QR.
    :return: Tipo de procesamiento seleccionado.
    """
    def select_handwritten():
        nonlocal selected_option
        selected_option = "handwritten"
        window.destroy()

    def select_computer():
        nonlocal selected_option
        selected_option = "computer"
        window.destroy()

    def select_qr_code():
        nonlocal selected_option
        selected_option = "qr_code"
        window.destroy()

    selected_option = None

    window = Tk()
    window.title("Seleccionar Tipo de Procesamiento")
    window.geometry("400x300")
    window.attributes("-topmost", True) 

    label = Label(window, text="Seleccione el tipo de procesamiento:", font=("Arial", 16))
    label.pack(pady=20)

    button_handwritten = Button(window, text="Texto Manuscrito", font=("Arial", 14), command=select_handwritten)
    button_handwritten.pack(pady=10)

    button_computer = Button(window, text="Texto de Ordenador", font=("Arial", 14), command=select_computer)
    button_computer.pack(pady=10)

    button_qr_code = Button(window, text="Código QR", font=("Arial", 14), command=select_qr_code)
    button_qr_code.pack(pady=10)

    window.mainloop()

    return selected_option

def show_text_window(predicted_text, title="Texto Predicho"):
    """
    Muestra una ventana con el texto predicho y permite continuar el programa al cerrarla.
    :param predicted_text: El texto predicho o decodificado.
    :param title: El título de la ventana.
    """
    window = Tk()
    window.title(title)
    window.geometry("600x400")
    window.attributes("-topmost", True) 
    window.update() 

    # Etiqueta de título
    label = Label(window, text=title, font=("Arial", 16, "bold"))
    label.pack(pady=10)

    # Área de texto con scroll
    text_area = Text(window, wrap="word", font=("Arial", 14))
    text_area.insert(END, predicted_text)
    text_area.config(state="disabled")  
    text_area.pack(expand=True, fill="both", padx=10, pady=10)

    scrollbar = Scrollbar(text_area, command=text_area.yview)
    text_area.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    window.mainloop()

def main():
    # Seleccionar imagen
    print("Seleccione una imagen desde el explorador de archivos...")
    image_path = select_image()
    if not image_path:
        print("No se seleccionó ninguna imagen. Cerrando.")
        return

    print(f"Imagen seleccionada: {image_path}")

    # Elegir opción de procesamiento
    print("Seleccione el tipo de procesamiento...")
    processing_option = choose_processing_option()
    if not processing_option:
        print("No se seleccionó ninguna opción. Cerrando.")
        return

    print(f"Opción seleccionada: {processing_option}")

    if processing_option == "qr_code":
        try:
            # Procesar código QR
            print("Procesando Código QR...")
            qr_code_image = extract_qr_code(image_path)
            decoded_data = decode_qr_code(qr_code_image)
            
            # Mostrar datos del QR en una ventana
            show_text_window(decoded_data, title="Datos del Código QR")

            # Mostrar imagen del QR Code detectado
            cv2.imshow("QR Code Detectado", qr_code_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except ValueError as e:
            print(e)

    else:
        # Elegir modelo de texto
        print("Seleccione el tipo de letra...")
        model_path = "models/cnn_model.keras" if processing_option == "handwritten" else "models/cnn_model_fonts.keras"
        print(f"Modelo seleccionado: {model_path}")

        # Segmentar caracteres
        segmented_dir = "data/segmented_chars"
        print("Segmentando caracteres...")
        segment_characters_with_lines(image_path, segmented_dir)

        # Predecir texto
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
