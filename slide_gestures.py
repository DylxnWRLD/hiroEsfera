import cv2, mediapipe as mp, numpy as np, pyautogui, time

# --- CONFIGURACIÓN DE PARÁMETROS ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Tiempo mínimo entre acciones para evitar pulsaciones múltiples (antirebote)
COOLDOWN_TIME =  2.0
# Pausa ligera para asegurar que PyAutoGUI registre la tecla
pyautogui.PAUSE = 0.05 
# ----------------------------------

def dedos_arriba(landmarks, w, h):
    # Convierte las coordenadas normalizadas a píxeles
    pts = np.array([[int(l.x*w), int(l.y*h)] for l in landmarks])
    # Inicializa la lista de dedos. El pulgar puede ser 1 (arriba), 0 (neutro), -1 (abajo)
    dedos = [0] * 5
    
    # Detección de Índice, Medio, Anular y Meñique (1 a 4)
    # Se levanta si la punta (tip) está más arriba (menor valor Y) que el pip (f-2)
    tips = [8, 12, 16, 20]
    for i, f in enumerate(tips):
        # f es la punta, f-2 es la articulación PIP (la del medio del dedo)
        dedos[i+1] = 1 if pts[f][1] < pts[f-2][1] else 0
        
    # Detección del Pulgar (dedos[0])
    # La articulación de referencia para el pulgar es la CMC/MP (punto 2)
    
    # 1. Pulgar Arriba (Like): La punta (4) está MÁS ARRIBA (menor valor Y) que la articulación base (2)
    if pts[4][1] < pts[2][1]:
        dedos[0] = 1 # Pulgar Arriba (Like)
    
    # 2. Pulgar Abajo (Dislike): La punta (4) está MÁS ABAJO (mayor valor Y) que la base (2)
    # Se añade un offset de 30px para asegurar un movimiento intencional
    elif pts[4][1] > pts[2][1] + 30: 
        dedos[0] = -1 # Pulgar Abajo (Dislike)
        
    return dedos 

# -------------------------------------------------------------------------------------
# --- BUCLE PRINCIPAL ---
# -------------------------------------------------------------------------------------

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    last_action_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        frame = cv2.flip(frame, 1) # Reflejar para visión tipo espejo
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_time = time.time()
        
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            d = dedos_arriba(lm, w, h)
            
            # Chequea si el cooldown ha pasado
            if current_time - last_action_time > COOLDOWN_TIME:
            
                # 1. GESTO: Pulgar Arriba (LIKE) -> AVANZAR (RIGHT)
                # Patrón: [1, 0, 0, 0, 0] (Pulgar Arriba, otros abajo)
                if d == [1, 0, 0, 0, 0]:
                    pyautogui.press('right')
                    last_action_time = current_time 
                    cv2.putText(frame, 'AVANZAR >> (LIKE)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 2. GESTO: Pulgar Abajo (DISLIKE) -> RETROCEDER (LEFT)
                # Patrón: [-1, 0, 0, 0, 0] (Pulgar Abajo, otros abajo)
                elif d == [-1, 0, 0, 0, 0]:
                    pyautogui.press('left')
                    last_action_time = current_time 
                    cv2.putText(frame, '<< RETROCEDER (DISLIKE)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Si el gesto se detecta, actualiza el cooldown
                
            cv2.putText(frame, f'Pulgar/Dedos: {d}', (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

        cv2.imshow("Slide Control (ESC para salir)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()