import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Default state if no hand is detected
    drawing_mode = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the full landmarks list
            lm = hand_landmarks.landmark
            
            # --- GESTURE DETECTION LOGIC ---
            # A finger is 'up' if its tip Y is smaller (higher up) than its PIP joint Y
            index_up  = lm[8].y  < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up   = lm[16].y < lm[14].y
            pinky_up  = lm[20].y < lm[18].y
            
            # Drawing mode activates ONLY when the index finger is up and all others are down
            drawing_mode = index_up and not middle_up and not ring_up and not pinky_up
            
            # Extract tracking point for landmark 8 (Index Tip)
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)
            points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # --- DISPLAY MODE TEXT ---
    if drawing_mode:
        text = "DRAWING"
        color = (0, 255, 0) # Green
    else:
        text = "NOT DRAWING"
        color = (0, 0, 255) # Red

    # cv2.putText(image, text, organization, font, fontScale, color, thickness)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
