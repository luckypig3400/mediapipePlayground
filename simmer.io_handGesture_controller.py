import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

desktop_width, desktop_height = pyautogui.size()
pyautogui.PAUSE = 0

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


def calculate_3_point_angle(pointAx: float, pointAy: float, centerPointX: float, centerPointY: float, pointCx: float,
                            pointCy: float) -> [float]:  # return angle
    # 公式參考自:
    # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
    vectorU = [pointAx - centerPointX, pointAy - centerPointY]  # 向量U
    vectorV = [pointCx - centerPointX, pointCy - centerPointY]  # 向量V
    lenghOfVectorU = math.sqrt((vectorU[0] * vectorU[0] + vectorU[1] * vectorU[1]))
    lenghOfVectorV = math.sqrt((vectorV[0] * vectorV[0] + vectorV[1] * vectorV[1]))
    innerProduct_of_UV = vectorU[0] * vectorV[0] + vectorU[1] * vectorV[1]  # 向量U 與 向量V的內積
    cosTheta = innerProduct_of_UV / (lenghOfVectorU * lenghOfVectorV)  # 向量U 與 向量V夾角的cosine值
    angle_in_degree = math.acos(cosTheta) * 180 / math.pi  # 弧度*180/pi轉成角度
    return angle_in_degree


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
        # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
        try:
            try:
                thumbAngle = calculate_3_point_angle(idx_to_coordinates[4][0], idx_to_coordinates[4][1],
                                                     idx_to_coordinates[3][0], idx_to_coordinates[3][1],
                                                     idx_to_coordinates[2][0], idx_to_coordinates[2][1])
                # print("thumbAngle:" + str(thumbAngle))
                indexFingerAngle = calculate_3_point_angle(idx_to_coordinates[5][0], idx_to_coordinates[5][1],
                                                           idx_to_coordinates[6][0], idx_to_coordinates[6][1],
                                                           idx_to_coordinates[7][0], idx_to_coordinates[7][1])
                middleFingerAngle = calculate_3_point_angle(idx_to_coordinates[9][0], idx_to_coordinates[9][1],
                                                            idx_to_coordinates[10][0], idx_to_coordinates[10][1],
                                                            idx_to_coordinates[11][0], idx_to_coordinates[11][1])
                ringFingerAngle = calculate_3_point_angle(idx_to_coordinates[13][0], idx_to_coordinates[13][1],
                                                          idx_to_coordinates[14][0], idx_to_coordinates[14][1],
                                                          idx_to_coordinates[15][0], idx_to_coordinates[15][1])
                pinkyAngle = calculate_3_point_angle(idx_to_coordinates[17][0], idx_to_coordinates[17][1],
                                                     idx_to_coordinates[18][0], idx_to_coordinates[18][1],
                                                     idx_to_coordinates[19][0], idx_to_coordinates[19][1])
            except:
                # thumbAngle = 180
                print("Oops found missing thumb point")

            if thumbAngle < 150:
                fingerBendStatus[0] = 1
                cv2.putText(image, "thumb bent, angle:" + str(thumbAngle), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (255, 255, 255), 2)
            else:
                fingerBendStatus[0] = 0

            if indexFingerAngle < 135:
                fingerBendStatus[1] = 1
                pyautogui.keyDown('1')
                cv2.putText(image, "index finger bent, angle:" + str(indexFingerAngle), (30, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[1] = 0
                pyautogui.keyUp('1')

            if middleFingerAngle < 135:
                fingerBendStatus[2] = 1
                pyautogui.keyDown('2')
                cv2.putText(image, "middle finger bent, angle:" + str(middleFingerAngle), (30, 90),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[2] = 0
                pyautogui.keyUp('2')

            if ringFingerAngle < 135:
                fingerBendStatus[3] = 1
                pyautogui.keyDown('3')
                cv2.putText(image, "ring finger bent, angle:" + str(ringFingerAngle), (30, 120),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            else:
                fingerBendStatus[3] = 0
                pyautogui.keyUp('3')

            if pinkyAngle < 135:
                fingerBendStatus[4] = 1
                pyautogui.keyDown('4')
                cv2.putText(image, "pinky bent, angle:" + str(pinkyAngle), (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (255, 255, 255), 2)
            else:
                fingerBendStatus[4] = 0
                pyautogui.keyUp('4')
        except:
            print("Oops found Missing Joints")
        # above is to judge if finger has bent

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
