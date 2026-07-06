import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Store all strokes
# -----------------------------
strokes = []
previous_drawing_mode = False

ERASER_RADIUS = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    drawing_mode = False
    eraser_mode = False

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            # -----------------------------
            # Finger states
            # -----------------------------
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # -----------------------------
            # Gestures
            # -----------------------------
            drawing_mode = (
                index_up and
                not middle_up and
                not ring_up and
                not pinky_up
            )

            eraser_mode = (
                index_up and
                middle_up and
                not ring_up and
                not pinky_up
            )

            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            # -----------------------------
            # Cursor
            # -----------------------------
            if eraser_mode:
                cv2.circle(
                    frame,
                    (cx, cy),
                    ERASER_RADIUS,
                    (255, 0, 0),
                    2
                )
            else:
                cv2.circle(
                    frame,
                    (cx, cy),
                    15,
                    (0, 255, 0),
                    cv2.FILLED
                )

            # -----------------------------
            # Start a new stroke
            # -----------------------------
            if drawing_mode:

                if not previous_drawing_mode:
                    strokes.append([])

                strokes[-1].append((cx, cy))

            # -----------------------------
            # Erase nearby points
            # -----------------------------
            if eraser_mode:

                for stroke in strokes[:]:

                    stroke[:] = [
                        pt
                        for pt in stroke
                        if (
                            (pt[0] - cx) ** 2 +
                            (pt[1] - cy) ** 2
                        ) > ERASER_RADIUS ** 2
                    ]

                    if len(stroke) < 2:
                        strokes.remove(stroke)

    previous_drawing_mode = drawing_mode

    # -----------------------------
    # Draw all strokes
    # -----------------------------
    for stroke in strokes:

        for i in range(len(stroke) - 1):

            cv2.line(
                frame,
                stroke[i],
                stroke[i + 1],
                (0, 0, 255),
                5
            )

    # -----------------------------
    # Status text
    # -----------------------------
    if drawing_mode:
        cv2.putText(
            frame,
            "DRAWING",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    elif eraser_mode:
        cv2.putText(
            frame,
            "ERASING",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

    else:
        cv2.putText(
            frame,
            "NOT DRAWING",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Hand Tracking Paint", frame)

    key = cv2.waitKey(1) & 0xFF


    if key == ord("c"):
        strokes.clear()


    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()