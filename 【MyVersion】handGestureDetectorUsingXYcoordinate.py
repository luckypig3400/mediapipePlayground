import cv2
import mediapipe as mp
import numpy as np
import pyautogui

desktop_width, desktop_height = pyautogui.size()


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
webcam_id = 1
window_name = 'Hand Gesture Detector'

hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)

cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow(window_name, cam_width, cam_height)
cv2.moveWindow(window_name, 300, 300)
fingerBendStatus = [0, 0, 0, 0, 0]  # 0~4 : thumb~pinky

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

        # below is to judge if finger has bent
        # TODO:focus on thumb bend accuracy and Three judge accuracy(use angle to judge if finger has bent)
        try:
            if idx_to_coordinates[4][1] + 36 > idx_to_coordinates[3][1] and idx_to_coordinates[3][0] < \
                    idx_to_coordinates[4][0] < idx_to_coordinates[20][0]:  # right hand
                fingerBendStatus[0] = 1
                # cv2.putText(image, "thumb bent(right hand)", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255),2)
            elif idx_to_coordinates[4][1] + 36 > idx_to_coordinates[3][1] and idx_to_coordinates[3][0] > \
                    idx_to_coordinates[4][0] > idx_to_coordinates[20][0]:  # left hand
                fingerBendStatus[0] = 1
                # cv2.putText(image, "thumb bent(left hand)", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[0] = 0

            if idx_to_coordinates[8][1] > idx_to_coordinates[5][1] or idx_to_coordinates[7][1] > \
                    idx_to_coordinates[5][1]:
                fingerBendStatus[1] = 1
                # cv2.putText(image, "index finger bent", (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[1] = 0

            if idx_to_coordinates[12][1] > idx_to_coordinates[9][1] or idx_to_coordinates[11][1] > \
                    idx_to_coordinates[9][1]:
                fingerBendStatus[2] = 1
                # cv2.putText(image, "middle finger bent", (30, 90), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[2] = 0

            if idx_to_coordinates[16][1] > idx_to_coordinates[13][1] or idx_to_coordinates[15][1] > \
                    idx_to_coordinates[13][1]:
                fingerBendStatus[3] = 1
                # cv2.putText(image, "ring finger bent", (30, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[3] = 0

            if idx_to_coordinates[20][1] > idx_to_coordinates[17][1] or idx_to_coordinates[19][1] > \
                    idx_to_coordinates[17][1]:
                fingerBendStatus[4] = 1
                # cv2.putText(image, "pinky bent", (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[4] = 0
        except:
            print("Oops found Missing Joints")
        # above is to judge if finger has bent

        # below is hand gesture judge
        if fingerBendStatus == [1, 1, 1, 1, 1]:
            cv2.putText(image, "Zero", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 0, 1, 1, 1]:
            cv2.putText(image, "One", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 0, 0, 1, 1]:
            cv2.putText(image, "Two", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 0, 0, 0, 1]:
            cv2.putText(image, "Three", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 0, 0, 0, 0]:
            cv2.putText(image, "Four", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 0, 0, 0, 0]:
            cv2.putText(image, "Five", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 1, 1, 1, 0]:
            cv2.putText(image, "Six", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 0, 1, 1, 1]:
            cv2.putText(image, "Seven", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 0, 0, 1, 1]:
            cv2.putText(image, "Eight", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 0, 0, 0, 1]:
            cv2.putText(image, "Nine", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 1, 0, 0, 0]:
            cv2.putText(image, "OK", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [0, 0, 1, 1, 0]:
            cv2.putText(image, "SpiderMan", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif fingerBendStatus == [1, 0, 1, 1, 0]:
            cv2.putText(image, "Rock", (int(cam_width/2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # above is hand gesture judge

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
