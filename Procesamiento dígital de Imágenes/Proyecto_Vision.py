import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


def main():
    print("--- Iniciando Proyecto de Visión ---")

    # 1. Configuración del modelo
    model_path = 'hand_landmarker.task'

    # Verificar si el archivo existe antes de seguir
    if not os.path.exists(model_path):
        print(f"ERROR: No se encuentra el archivo '{model_path}'")
        print(f"Buscando en: {os.getcwd()}")
        print("Por favor, descarga el archivo .task y ponlo en esta carpeta.")
        return

    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("Detector de MediaPipe cargado exitosamente.")
    except Exception as e:
        print(f"Error al inicializar el detector: {e}")
        return

    # 2. Configurar Cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    print("Cámara lista. Presiona ESC para salir.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Detectar
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                dedos = []
                # Pulgar (comparación horizontal)
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    dedos.append(1)
                else:
                    dedos.append(0)

                # Otros 4 dedos (comparación vertical: punto 8 vs 6, 12 vs 10, etc.)
                puntas = [8, 12, 16, 20]
                bases = [6, 10, 14, 18]
                for p, b in zip(puntas, bases):
                    if hand_landmarks[p].y < hand_landmarks[b].y:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                total_dedos = dedos.count(1)
                cv2.putText(frame, f'Dedos: {total_dedos}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 4. Dibujar puntos y conexiones manualmente
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Dibujar los 21 puntos (Landmarks)
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Dibujar conexiones básicas (opcional, para que se vea mejor)
                puntos = hand_landmarks
                # Lista de conexiones de los dedos (índices de los puntos)
                conexiones = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
                    (5, 9), (9, 13), (13, 17)  # Palma
                ]
                for start, end in conexiones:
                    x1, y1 = int(puntos[start].x * frame.shape[1]), int(puntos[start].y * frame.shape[0])
                    x2, y2 = int(puntos[end].x * frame.shape[1]), int(puntos[end].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 5. Mostrar resultado
        cv2.imshow('MediaPipe Vision Nativa', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Programa terminado ---")


if __name__ == "__main__":
    main()