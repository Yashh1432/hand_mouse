import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time  # Added to fix NameError

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()  

# Scroll state
scroll_mode = False

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Index finger tip (landmark 8) for cursor
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)

            # Map to screen coordinates
            screen_x = np.interp(x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(y, [0, frame_height], [0, screen_height])

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Thumb tip (landmark 4) for clicks
            thumb_tip = hand_landmarks.landmark[4]
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            # Middle finger tip (landmark 12) for right click and scrolling
            middle_finger_tip = hand_landmarks.landmark[12]
            middle_x = int(middle_finger_tip.x * frame_width)
            middle_y = int(middle_finger_tip.y * frame_height)

            # Wrist (landmark 0) and Ring/Pinky tips for reference
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * frame_width)
            wrist_y = int(wrist.y * frame_height)
            ring_finger_tip = hand_landmarks.landmark[16]
            ring_x = int(ring_finger_tip.x * frame_width)
            ring_y = int(ring_finger_tip.y * frame_height)
            pinky_tip = hand_landmarks.landmark[20]
            pinky_x = int(pinky_tip.x * frame_width)
            pinky_y = int(pinky_tip.y * frame_height)

            # Calculate distances
            distance_thumb_index = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)
            distance_thumb_middle = np.sqrt((middle_x - thumb_x)**2 + (middle_y - thumb_y)**2)

            # Debug: Print key positions
            print(f"Wrist: ({wrist_x}, {wrist_y}), Index: ({x}, {y}), Distance Y: {abs(wrist_y - y)}")

            # Left click if thumb and index finger are close
            if distance_thumb_index < 50:
                pyautogui.click()
                cv2.putText(frame, "Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right click if thumb and middle finger are close
            if distance_thumb_middle < 30:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click!", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Toggle scroll mode with close wrist gesture (index tip below wrist by a threshold)
            if abs(wrist_y - y) > 100 and wrist_y < y:  # Index tip significantly below wrist (hand bent upward)
                scroll_mode = not scroll_mode
                time.sleep(0.5)  # Debounce to avoid rapid toggling
                cv2.putText(frame, f"Scroll Mode: {'ON' if scroll_mode else 'OFF'}", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                print(f"Scroll Mode toggled to: {scroll_mode}")  # Debug scroll mode state

            # Scroll if scroll mode is on and index is close to middle
            if scroll_mode:
                proximity_x = abs(x - middle_x)
                proximity_y = abs(y - middle_y)
                if proximity_x < 30 and proximity_y < 30:  # Relaxed threshold
                    if 'prev_y' not in globals():
                        prev_y = y
                    else:
                        prev_y = globals()['prev_y']
                    dy = prev_y - y
                    globals()['prev_y'] = y
                    if abs(dy) > 20:  # Reduced sensitivity threshold
                        scroll_amount = dy // 5  # Adjusted scroll speed
                        pyautogui.scroll(scroll_amount)
                        cv2.putText(frame, f"Scroll: {scroll_amount}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        print(f"Scrolling: dy={dy}, scroll_amount={scroll_amount}")  # Debug scroll values

    # Display frame
    cv2.imshow("Hand Gesture Mouse Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()