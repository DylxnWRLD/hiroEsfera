import cv2
import mediapipe as mp

# Inicializa el Mediapipe para la deteccion de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicializa la captura de video desde la camara web
cap = cv2.VideoCapture(0)

print("Iniciando la interfaz gestual. Presiona 'q' para salir.")
# Se hace un bucle para procesar cada frame del video para detectar las manos
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Voltear la imagen horizontalmente para una vista de selfie
    image = cv2.flip(image,1)

    # Convierte la imagen a RGB, que es el formato que usa Mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar las manos
    results = hands.process(image_rgb)

    # Dibujar los puntos de referencia de la mano si se detectan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener las coordenadas del centro de la mano (punto de referencia 9, la muñeca)
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            center_x = int(sum(x_coords) / len(x_coords) * image.shape[1])
            center_y = int(sum(y_coords) / len(y_coords) * image.shape[0])

            # Mostrar las coordenadas en la pantalla
            cv2.putText(image, f"Mano detectada en ({center_x}, {center_y})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Ejemplo de lógica gestual simple:
            # Si el centro de la mano está en la parte superior de la pantalla, haz algo
            if center_y < image.shape[0] * 0.3:
                cv2.putText(image, "Gesto: Mano en alto", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow("Interfaz Gestual", image)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()