import cv2

cm = cv2.VideoCapture(0)

if not cm.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cm.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Original
    original = frame

    # Región otro color
    roi_img = frame.copy()
    h, w, _ = frame.shape

    x1 = int(w * 0.3)
    y1 = int(h * 0.3)

    # Tamaño del rectángulo (30% del frame)
    roi_width = int(w * 0.3)
    roi_height = int(h * 0.3)

    # Esquina inferior derecha
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    roi_img[y1:y2, x1:x2] = (0, 255, 0)
    cv2.rectangle(roi_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Imagen con diferente tamaño
    resized = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # Mostrar ventanas
    cv2.imshow("1 - Imagen Original", original)
    cv2.imshow("2 - Region en otro color", roi_img)
    cv2.imshow("3 - Escala de Grises", gray)
    cv2.imshow("4 - Imagen Redimensionada", resized)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cm.release()
cv2.destroyAllWindows()
