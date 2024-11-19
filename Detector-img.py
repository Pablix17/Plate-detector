import cv2
import numpy as np
import pytesseract
import glob

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseract_config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_char_blacklist=@#$%&*()'

def detectar_color_amarillo(image):
    """
    Detecta regiones amarillas en la imagen usando HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rango de color amarillo en HSV
    lower_yellow = np.array([10, 100, 100], dtype="uint8")
    upper_yellow = np.array([40, 255, 255], dtype="uint8")

    # Crear máscara para áreas amarillas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def aumentar_tamano(image, factor=4):
    """
    Amplía la imagen en un factor dado.
    """
    width = int(image.shape[1] * factor)
    height = int(image.shape[0] * factor)
    dim = (width, height)
    return cv2.resize(image, dim)

def aplicar_filtro_enfoque(image):
    """
    Aplica un filtro de enfoque a la imagen usando un kernel de paso alto.
    """
    kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def subir_calidad_imagen(image):
    """
    Mejora la calidad de la imagen mediante aumento de resolución y mejora de contraste.
    """
    high_res_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Mejorar el contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    high_contrast_image = clahe.apply(cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY))

    # Convertir la imagen de vuelta a BGR para el enfoque posterior
    high_contrast_image = cv2.cvtColor(high_contrast_image, cv2.COLOR_GRAY2BGR)
    
    # Aplicar un enfoque adicional
    return aplicar_filtro_enfoque(high_contrast_image)

def preprocesar_placa(image):
    """
    Preprocesa la imagen de la placa: mejora bordes, aplica desenfoque, etc.
    """
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Suavizar la imagen para reducir ruido
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(gray_image, 100, 200)
    
    return edges

def procesar_imagen_con_texto(image, x, y, w, h):
    """
    Procesa una región seleccionada de la imagen para mejorar la calidad, 
    aumentar la resolución y leer texto con OCR.
    """
    # Recortar la región de la placa
    placa = image[y:y+h, x:x+w]

    # Subir la calidad de la imagen
    placa_calidad = subir_calidad_imagen(placa)

    # Leer el texto de la placa en alta resolución
    text = pytesseract.image_to_string(placa_calidad, config=tesseract_config).strip()

    return text, placa_calidad

def detectar_y_leer_placas(image_path):
    """
    Detecta placas basadas en el color amarillo y extrae texto usando OCR.
    """
    # Cargar imagen original
    image = cv2.imread(image_path)
    if image is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return []

    # Aumentar el tamaño de la imagen
    image = aumentar_tamano(image, factor=1.5)

    # Detectar regiones amarillas
    mask_yellow = detectar_color_amarillo(image)
    cv2.imshow("Máscara Amarilla", mask_yellow)

    # Mejorar bordes con Canny
    edges = cv2.Canny(mask_yellow, 100, 200)
    cv2.imshow("Bordes Mejorados", edges)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    detected_texts = []

    largest_contour = None
    largest_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)

        # Filtrar las áreas que no parecen placas (relación de aspecto similar a las placas)
        if aspect_ratio > 0.8 and aspect_ratio < 4.0 and area > 200:
            if area > largest_area:  # Buscar el área más grande
                largest_area = area
                largest_contour = (x, y, w, h)

    if largest_contour:
        x, y, w, h = largest_contour
        # Procesar la región de la placa para leer el texto
        text, placa_calidad = procesar_imagen_con_texto(image, x, y, w, h)

        if text:
            detected_texts.append(text)
            print(f"Texto detectado: {text}")
            
            # Dibujar el texto detectado sobre la imagen
            cv2.putText(output_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            # Dibujar el contorno de la placa en la imagen original
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Placa Detectada", placa_calidad)

    # Mostrar la imagen con el texto y la placa
    cv2.imshow("Detección de Placas", output_image)

    # Esperar a que se presione una tecla para cerrar todas las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_texts

def procesar_imagenes_con_glob():
    """
    Procesa todas las imágenes .jpg en el directorio usando glob.
    """
    image_files = glob.glob('*.jpg')  # Asegúrate de que las imágenes estén en el mismo directorio

    for image_path in image_files:
        print(f"Procesando imagen: {image_path}")
        textos_detectados = detectar_y_leer_placas(image_path)
        print(f"Textos detectados en {image_path}: {textos_detectados}")

# Llamar a la función para procesar las imágenes
procesar_imagenes_con_glob()
