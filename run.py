import cv2
import mediapipe as mp
import anvil.server
import signal
import sys


def add_entry():
    anvil.server.call("add_entry")


def cleanup():
    anvil.server.disconnect()
    sys.exit(0)


def signal_handler(sig, frame):
    cleanup()


signal.signal(signal.SIGINT, signal_handler)
anvil.server.connect("[UPLINK KEY]")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

counter = 0
started = False
down = False
nose_init = 0
nose_z = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cleanup()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id == 0: # 0-nose
                # z approaches 0 as you move away from the camera
                nose_z = abs(lm.z)
                if not started:
                    started = True
                    nose_init = nose_z
                d = abs(nose_z - nose_init)
                print(f"{counter} {d}")
                if d > 0.8 and not down:
                    down = True
                if down and d < 0.8:
                    counter+=1
                    down = False
                    add_entry()

            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        cleanup()
