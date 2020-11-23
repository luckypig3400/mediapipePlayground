import cv2
import mediapipe as mp
import pyautogui

webcam_id = 1
window_name = 'Hand Tracking Control Mouse Example'
desktop_width, desktop_height = pyautogui.size()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(hand_landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_px = mp.python.solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                                             image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
                if idx == 8:
                    print(idx, ":", idx_to_coordinates[idx])

        # Screen Monitor
        window_x, window_y, window_w, window_h = cv2.getWindowImageRect(window_name)
        # xpos, ypos = kp_orig[4,:2]
        xpos, ypos = idx_to_coordinates[8]
        pyautogui.moveTo(xpos / window_w * desktop_width, ypos / window_h * desktop_height)
        # x = xpos / window_w * desktop_width
        # y = ypos / window_h * desktop_height

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == 27:
        print(idx_to_coordinates)
        break

hands.close()
cap.release()
