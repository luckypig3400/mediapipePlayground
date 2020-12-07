import cv2
import mediapipe as mp

import numpy as np
import pyautogui

webcam_id = 1
window_name = 'HandsTracking_with_3Dimensional_Coordinates_Output_by_PTChen'

desktop_width, desktop_height = pyautogui.size()
pyautogui.PAUSE = 0
ratioxy = 5


def normalized_2_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> [int, int]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or np.isclose(0, value)) and (value < 1 or np.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return [None, None]
    x_px = min(np.floor(normalized_x * image_width), image_width - 1)
    y_px = min(np.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def normalized_3_pixel_coordinates(
        normalized_x: float, normalized_y: float, normalized_z: float, image_width: int,
        image_height: int) -> [float, float, float]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or np.isclose(0, value)) and (value < 1 or np.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return [None, None, None]
    x_px = min(normalized_x * image_width, image_width - 1)
    y_px = min(normalized_y * image_height, image_height - 1)
    z_px = normalized_z * 1000  # 100cm = 1000mm
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
cv2.moveWindow(window_name, 300, 300)

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
            landmark_px = normalized_3_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates.append(landmark_px)
        # print("Before np.array Method:", idx_to_coordinates)
        idx_to_coordinates = np.array(idx_to_coordinates)
        # print("After np.array Method:", idx_to_coordinates)

        for i in range(5, 9):  # print index finger joints(5~9) x,y,z coordinates
            try:
                singleJointInfo = "x:" + str(int(idx_to_coordinates[i][0])) + " y:" + str(int(
                    idx_to_coordinates[i][1])) + " z:" + str(idx_to_coordinates[i][2])
                textLocation = (int(idx_to_coordinates[i][0]), int(idx_to_coordinates[i][1]))

                cv2.putText(image, singleJointInfo, textLocation, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

                if idx_to_coordinates[7][1] > idx_to_coordinates[5][1] or idx_to_coordinates[8][1] > \
                        idx_to_coordinates[5][1]:
                    cv2.putText(image, "index finger bent", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            except:
                print("Oops found Missing Joints")

        # print("idx_to_coordinates info:dtype:" + str(idx_to_coordinates.dtype) + "\tshape:" + str(
        #     idx_to_coordinates.shape) + "\tsize:" + str(idx_to_coordinates.size))

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

# INDEX_FINGER_TIP = <HandLandmark.INDEX_FINGER_TIP: 8>
# INDEX_FINGER_DIP = <HandLandmark.INDEX_FINGER_DIP: 7>
# INDEX_FINGER_PIP = <HandLandmark.INDEX_FINGER_PIP: 6>
# INDEX_FINGER_MCP = <HandLandmark.INDEX_FINGER_MCP: 5>

# MIDDLE_FINGER_TIP = <HandLandmark.MIDDLE_FINGER_TIP: 12>
# MIDDLE_FINGER_DIP = <HandLandmark.MIDDLE_FINGER_DIP: 11>
# MIDDLE_FINGER_PIP = <HandLandmark.MIDDLE_FINGER_PIP: 10>
# MIDDLE_FINGER_MCP = <HandLandmark.MIDDLE_FINGER_MCP: 9>

# PINKY_TIP = <HandLandmark.PINKY_TIP: 20>
# PINKY_DIP = <HandLandmark.PINKY_DIP: 19>
# PINKY_PIP = <HandLandmark.PINKY_PIP: 18>
# PINKY_MCP = <HandLandmark.PINKY_MCP: 17>

# RING_FINGER_TIP = <HandLandmark.RING_FINGER_TIP: 16>
# RING_FINGER_DIP = <HandLandmark.RING_FINGER_DIP: 15>
# RING_FINGER_PIP = <HandLandmark.RING_FINGER_PIP: 14>
# RING_FINGER_MCP = <HandLandmark.RING_FINGER_MCP: 13>

# THUMB_CMC = <HandLandmark.THUMB_CMC: 1>
# THUMB_IP = <HandLandmark.THUMB_IP: 3>
# THUMB_MCP = <HandLandmark.THUMB_MCP: 2>
# THUMB_TIP = <HandLandmark.THUMB_TIP: 4>
#
# WRIST = <HandLandmark.WRIST: 0>
