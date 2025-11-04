import cv2
import mediapipe as mp
import math
import time

# --- Configuración de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) 
mp_draw = mp.solutions.drawing_utils

# --- Constantes y Puntos Clave ---
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP # CORREGIDO: Usando PINKY_TIP

# Articulaciones (PIP y IP)
INDEX_PIP = mp_hands.HandLandmark.INDEX_FINGER_PIP
MIDDLE_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
RING_PIP = mp_hands.HandLandmark.RING_FINGER_PIP
PINKY_PIP = mp_hands.HandLandmark.PINKY_PIP # CORREGIDO: Usando PINKY_PIP
THUMB_IP = mp_hands.HandLandmark.THUMB_IP 

# Grupos de dedos para la lógica de gesto
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# --- Inicialización de Video ---
cap = cv2.VideoCapture(0)
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480 
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

print("Iniciando la interfaz gestual. Cierra el puño para salir. Pulgar arriba para 'LIKE'.")


# --- Funciones de Detección de Gestos ---

def is_fist_closed(hand_landmarks):
    """Detecta el puño cerrado para salir (todos los dedos flexionados)."""
    
    # Los 4 dedos (índice a meñique) deben estar doblados (punta Y > PIP Y)
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            return False
            
    # El pulgar debe estar doblado (punta Y > IP Y)
    if hand_landmarks.landmark[THUMB_TIP].y < hand_landmarks.landmark[THUMB_IP].y:
         return False
         
    return True

def is_thumbs_up(hand_landmarks):
    """Detecta el gesto de 'Pulgar Arriba' (Like)."""
    
    # 1. El pulgar debe estar EXTENDIDO hacia ARRIBA (punta Y < IP Y)
    thumb_is_up = hand_landmarks.landmark[THUMB_TIP].y < hand_landmarks.landmark[THUMB_IP].y
    
    # 2. Los 4 dedos restantes deben estar FLEXIONADOS (punta Y > PIP Y)
    all_fingers_down = True
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            all_fingers_down = False
            break
            
    # Es un "Like" solo si el pulgar está arriba Y los demás dedos están abajo.
    return thumb_is_up and all_fingers_down


# --- Bucle Principal ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # 1. Preprocesamiento de la imagen
    image = cv2.flip(image, 1) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    gesture_text = "Esperando Gesto..."
    exit_program = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos y conexiones de la mano
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- LÓGICA DE DETECCIÓN DE GESTOS ---
            
            if is_fist_closed(hand_landmarks):
                # Prioridad 1: Gesto para salir (Puño Cerrado)
                gesture_text = "¡PUÑO CERRADO! CERRANDO PROGRAMA..."
                cv2.putText(image, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                exit_program = True
                break
                
            elif is_thumbs_up(hand_landmarks):
                # Prioridad 2: Gesto de Like (Pulgar Arriba)
                gesture_text = "👍 ¡GESTO DETECTADO: LIKE! 👍"
                cv2.putText(image, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            else:
                gesture_text = "Mano Abierta o Gesto No Reconocido"
                cv2.putText(image, gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # (Opcional) Mostrar centro de la mano
            x_coords = [l.x for l in hand_landmarks.landmark]
            y_coords = [l.y for l in hand_landmarks.landmark]
            center_x = int(sum(x_coords) / len(x_coords) * CAMERA_WIDTH)
            center_y = int(sum(y_coords) / len(y_coords) * CAMERA_HEIGHT)
            cv2.putText(image, f"Mano en ({center_x}, {center_y})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    # Lógica de salida
    if exit_program:
        cv2.imshow("Interfaz Gestual", image)
        cv2.waitKey(1000) # Muestra el mensaje de cierre por 1 segundo
        break

    # 2. Mostrar el resultado
    cv2.imshow("Interfaz Gestual", image)

    # 3. Salida por tecla
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Liberar recursos
cap.release()
cv2.destroyAllWindows()