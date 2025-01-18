# **TFA OCR: Reconocimiento Óptico de Caracteres - Guzmán G Riancho**

TFA OCR es un proyecto que implementa un sistema de reconocimiento óptico de caracteres (OCR) utilizando redes neuronales convolucionales. El proyecto permite segmentar caracteres de una imagen, incluyendo espacios y líneas múltiples, y predecir el texto presente en la imagen.

---

## **Entrenamiento y Preprocesamiento**

### **1. Preprocesamiento de datos (`preprocessing.py`)**

Este script organiza y prepara las imágenes para que el modelo las pueda usar.

- **Entrada**: Imágenes en `data/raw/alumno/`, separadas en:

  - `Mayusculas/`: Letras mayúsculas.
  - `Minusculas/`: Letras minúsculas.
  - `Numeros/`: Números.

- **Proceso**:

  1. Convierte las imágenes a escala de grises y las redimensiona a 32x32 píxeles.
  2. Genera variaciones (rotaciones y desplazamientos) para mejorar el modelo.

- **Salida**: Las imágenes procesadas y aumentadas se guardan en `data/unified/`.

---

### **2. Generación de un dataset con fuentes TTF (`generate_dataset.py`)**

Este script crea un nuevo dataset basado en fuentes tipográficas de computadora.

- **Entrada**: Archivos TTF de fuentes almacenados en `assets/fonts/`.
- **Proceso**:
  1. Se renderizan caracteres (mayúsculas, minúsculas y números) utilizando las fuentes TTF.
  2. Los caracteres renderizados se escalan y se centran según su contorno real.
  3. El dataset final se guarda en `data/unifiedFonts/` para ser usado en el entrenamiento.

---

### **3. Entrenamiento del modelo (`train.py`)**

Este script entrena el modelo para reconocer caracteres.

- **Entrada**: Imágenes de `data/unified/` o `data/unifiedFonts/`.
- **Modelo**: Red neuronal que aprende a distinguir letras y números.
- **Proceso**:

  1. Divide los datos en entrenamiento (80%) y validación (20%).
  2. Usa técnicas para mejorar el modelo como:
     - **Aumentación de datos**: Genera más datos en tiempo real.
     - **Early stopping**: Detiene el entrenamiento cuando no hay mejoras.

- **Salida**:
  - El modelo entrenado se guarda en:
    - `models/cnn_model.keras` (manuscritos).
    - `models/cnn_model_fonts.keras` (fuentes tipográficas).
  - Un archivo `label_to_index.npy` que indica qué etiqueta corresponde a cada carácter.

---

### **4. Evaluación del modelo (`evaluate.py`)**

Este script verifica cómo de bien funciona el modelo.

- Usa imágenes de prueba en `data/examples/`. Estas imágenes incluyen caracteres manuscritos y tipográficos, generados automáticamente.
- Genera un reporte con:
  - El porcentaje de aciertos.
  - Métricas como precisión, recall y F1-score.

Esto permite evaluar qué tan bien el modelo reconoce caracteres estilizados y fuentes tipográficas.

---

## **Instalación**

### 1. Clonar el repositorio

```bash
git clone https://github.com/guzmangrianchoUNEAT/TFA-OCR.git
cd TFA-OCR
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

```bash
.venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## **Estructura del proyecto**

El proyecto tiene la siguiente estructura:

```
.
├── data/
│   ├── input/           # Carpeta para colocar imágenes de entrada
│   ├── segmented_chars/ # Carpeta donde se guardan caracteres segmentados
│
├── models/
│   ├── cnn_model.keras       # Modelo CNN entrenado para manuscritos
│   ├── cnn_model_fonts.keras # Modelo CNN entrenado para fuentes tipográficas
│   ├── label_to_index.npy    # Diccionario de etiquetas del modelo
│
├── src/
│   ├── main.py                 # Archivo principal
│   ├── segment_characters.py   # Segmenta caracteres, detecta espacios y cambios de línea
│   ├── predict_word.py         # Predice texto
│   ├── generate_dataset.py     # Crea un dataset a partir de fuentes TTF
│
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación del proyecto
```

---

## **Cómo probar el proyecto**

### **1. Segmentar caracteres manualmente**

Puedes usar `segment_characters.py` para segmentar los caracteres de una imagen y guardarlos en la carpeta `data/segmented_chars`.

#### **Instrucciones**:

1. Coloca una imagen en la carpeta `data/input/` (por ejemplo, `prueba.png`).
2. Modifica la variable `input_image` en el archivo `segment_characters.py` para apuntar a tu imagen:
   ```python
   input_image = "data/input/prueba.png"
   ```
3. Ejecuta el script:
   ```bash
   python src/segment_characters.py
   ```
4. Los caracteres segmentados se guardarán en `data/segmented_chars`.

---

### **2. Usar el sistema completo con `main.py`**

El archivo `main.py` automatiza el flujo completo: selección de imagen, segmentación, predicción y visualización del texto.

#### **Instrucciones**:

1. Ejecuta el script principal:
   ```bash
   python src/main.py
   ```
2. Aparecerá un cuadro de diálogo para seleccionar una imagen desde tu explorador de archivos. Elige una imagen de la carpeta `data/input`.
3. El sistema segmentará los caracteres, detectará espacios y predecirá el texto presente en la imagen.
4. El texto predicho se mostrará en una ventana emergente profesional, y luego también en la consola.

---

## **Personalización**

### Espacios y saltos de línea

Puedes ajustar el umbral de detección de espacios y líneas modificando las variables `space_threshold` y `vertical_threshold` en `segment_characters_with_lines` del archivo `segment_characters.py`.

---
