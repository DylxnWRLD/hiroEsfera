import cv2

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
