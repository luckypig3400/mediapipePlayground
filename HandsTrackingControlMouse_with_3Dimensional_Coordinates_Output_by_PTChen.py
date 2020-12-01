import cv2
import mediapipe as mp

import numpy as np
import pyautogui

webcam_id = 1
window_name = 'Hand Tracking'

desktop_width, desktop_height = pyautogui.size()
pyautogui.PAUSE = 0
ratioxy = 5


def normalized_2_pixel_coordinates(
        normalized_x: float, normalized_y: float, normalized_z: float, image_width: int,
        image_height: int) -> [int, int, int]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or np.isclose(0, value)) and (value < 1 or
                                                        np.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return [None, None, None]
    x_px = min(np.floor(normalized_x * image_width), image_width - 1)
    y_px = min(np.floor(normalized_y * image_height), image_height - 1)
    z_px = np.floor(normalized_y * image_height)
    return x_px, y_px, z_px


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)


MouseL_Click = False
MouseR_Click = False


cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)

cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow(window_name, cam_width, cam_height)
cv2.moveWindow(window_name, 0, 0)

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
        idx_to_coordinates = []
        for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_px = normalized_2_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates.append(landmark_px)
        idx_to_coordinates = np.array(idx_to_coordinates)

        # Screen Monitor
        window_x, window_y, window_w, window_h = cv2.getWindowImageRect(window_name)
        # xpos, ypos = np.mean(idx_to_coordinates[:, :2], axis=0)
        not_None_px = idx_to_coordinates[idx_to_coordinates != None].reshape(-1,3)
        xpos, ypos = not_None_px[:, :2].mean(axis=0)
        # pyautogui.moveTo(window_x + xpos, window_y + ypos)
        # x = xpos / window_w * desktop_width
        # y = ypos / window_h * desktop_height
        wb, hb = window_w / ratioxy, window_h / ratioxy
        # x = np.min([np.max([1, xpos - wb]), window_w * (ratioxy-1)/ratioxy ])
        x = (xpos - wb) / ((ratioxy - 2)/ratioxy * window_w) * desktop_width
        # y = np.min([np.max([1, ypos - hb]), window_h * (ratioxy - 1) / ratioxy])
        y = (ypos - hb) / ((ratioxy - 2) / ratioxy * window_h) * desktop_height
        x = np.min([np.max([1, x]), desktop_width-1])
        y = np.min([np.max([1, y]), desktop_height-1])
        pyautogui.moveTo(x, y)

        # Mouse Control
        # Click_Threshold = np.linalg.norm(idx_to_coordinates[4, :] - idx_to_coordinates[2, :])
        Click_Threshold = 50
        if np.all(idx_to_coordinates[8]) and np.all(idx_to_coordinates[4]):
            MouseL = np.linalg.norm(idx_to_coordinates[8] - idx_to_coordinates[4])
        if np.all(idx_to_coordinates[12]) and np.all(idx_to_coordinates[4]):
            MouseR = np.linalg.norm(idx_to_coordinates[12] - idx_to_coordinates[4])

        if MouseL <= Click_Threshold and MouseL_Click is False:
            pyautogui.mouseDown(button='left')
            MouseL_Click = True
        elif MouseL > Click_Threshold and MouseL_Click is True:
            pyautogui.mouseUp(button='left')
            MouseL_Click = False
        else:
            pass

        # if MouseR <= Click_Threshold and MouseR_Click is False:
        #     pyautogui.mouseDown(button='right')
        #     MouseR_Click = True
        # elif MouseR > Click_Threshold and MouseR_Click is True:
        #     pyautogui.mouseUp(button='right')
        #     MouseR_Click = False
        # else:
        #     pass

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
