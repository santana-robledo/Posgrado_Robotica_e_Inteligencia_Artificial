import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

# BRAZO
j1 = sim.getObject('/PhantomXPincher/joint')
j2 = sim.getObject('/PhantomXPincher/link/joint')

# GRIPPER
gripper = sim.getObject('/PhantomXPincher/gripperClose_joint')

sim.startSimulation()

print("Brazo conectado")
model_path = 'hand_landmarker.task'

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("Sistema iniciado")

def contar_dedos(hand_landmarks):

    dedos = []

    # pulgar
    if hand_landmarks[4].x > hand_landmarks[3].x:
        dedos.append(1)
    else:
        dedos.append(0)

    # otros dedos
    puntas = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]

    for p, b in zip(puntas, bases):

        if hand_landmarks[p].y < hand_landmarks[b].y:
            dedos.append(1)
        else:
            dedos.append(0)

    return dedos.count(1)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:

        hand_landmarks = detection_result.hand_landmarks[0]

        total_dedos = contar_dedos(hand_landmarks)

        for landmark in hand_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        conexiones = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        for start, end in conexiones:

            x1 = int(hand_landmarks[start].x * frame.shape[1])
            y1 = int(hand_landmarks[start].y * frame.shape[0])

            x2 = int(hand_landmarks[end].x * frame.shape[1])
            y2 = int(hand_landmarks[end].y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255,255,255),
                2
            )

        wrist = hand_landmarks[0]

        x = wrist.x
        y = wrist.y

        angle1 = (x - 0.5) * 180
        angle2 = (0.5 - y) * 120

        angle1 = max(-90, min(90, angle1))
        angle2 = max(-45, min(45, angle2))

        if total_dedos == 3:

            sim.setJointTargetPosition(
                j1,
                math.radians(angle1)
            )

            sim.setJointTargetPosition(
                j2,
                math.radians(angle2)
            )

        elif total_dedos == 0:

            sim.setJointTargetPosition(j1, math.radians(0))  # base recta

            sim.setJointTargetPosition(j2, math.radians(-90))  # brazo hacia adelante

            sim.setJointTargetPosition(gripper, 0)

        elif total_dedos == 5:

            sim.setJointTargetPosition(j1, math.radians(0))

            sim.setJointTargetPosition(j2, math.radians(0))

            sim.setJointTargetPosition(gripper, 0.03)


    cv2.imshow("Control Gestual Robotico", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()

sim.stopSimulation()

print("Programa terminado")