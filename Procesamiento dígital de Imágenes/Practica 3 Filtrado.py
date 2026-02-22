import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 = cámara principal

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convierto a escala de grises

    # Filtro Gaussiano
    suavizado = cv2.GaussianBlur(gray, (15, 15), 0)

    #Detección de bordes
    edges = cv2.Canny(suavizado, 50, 150) #Umbral bajo, umbral alto

    cv2.imshow("Original", frame)
    cv2.imshow("Suavizado", suavizado)
    cv2.imshow("Bordes", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()