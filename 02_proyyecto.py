import cv2
import numpy as np
import pytesseract

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def detectar_color_amarillo(image):
    """
    Detecta regiones amarillas en la imagen usando HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir rango de color amarillo en HSV
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    # Crear máscara para áreas amarillas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def preprocesar_placa(placa):
    """
    Aplica preprocesamiento a la región de la placa antes del OCR.
    """
    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    
    # Mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # Aplicar binarización
    _, binary = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def aplicar_transformacion_afin(image, contour):
    """
    Aplica una transformación afín a la región de la placa para corregir la perspectiva.
    """
    # Obtener el rectángulo delimitador de la región amarilla
    x, y, w, h = cv2.boundingRect(contour)
    
    # Definir las esquinas originales de la región detectada
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h]])  # Esquinas superiores e inferiores
    pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])  # Nuevo rectángulo "plano"
    
    # Obtener la matriz de transformación afín
    matrix = cv2.getAffineTransform(pts1, pts2)
    
    # Aplicar la transformación afín
    transformed = cv2.warpAffine(image, matrix, (w, h))
    
    return transformed

def detectar_y_leer_placas_en_tiempo_real():
    """
    Detecta placas basadas en el color amarillo y extrae texto usando OCR en tiempo real desde la cámara.
    """
    # Inicializar la cámara (cámara predeterminada)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return
    
    while True:
        # Capturar un fotograma
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar fotograma.")
            break

        # Detectar regiones amarillas
        mask_yellow = detectar_color_amarillo(frame)

        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = frame.copy()
        detected_texts = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5 and w * h > 500:  # Filtrar posibles placas
                # Dibujar contorno azul en la imagen original
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Aplicar transformación afín a la región detectada
                placa_transformada = aplicar_transformacion_afin(frame, contour)

                # Preprocesar la placa transformada
                placa_bin = preprocesar_placa(placa_transformada)

                # Aplicar OCR
                text = pytesseract.image_to_string(placa_bin, config=tesseract_config).strip()
                if text:
                    detected_texts.append(text)
                    # Superponer el texto encima de la región de la placa
                    cv2.putText(output_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Mostrar la imagen con contornos dibujados y texto detectado
        cv2.imshow("Detección de Placas", output_image)
        
        # Si se presiona 'q', salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para detección en tiempo real
detectar_y_leer_placas_en_tiempo_real()
