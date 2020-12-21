import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

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
hand1_fingerBendStatus = [0, 0, 0, 0, 0]  # 0~4 : thumb~pinky
hand2_fingerBendStatus = [0, 0, 0, 0, 0]  # 0~4 : thumb~pinky
hand1_label = ""
hand2_label = ""
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


def calculate_3_point_angle(pointAx: float, pointAy: float,
                            centerPointX: float, centerPointY: float,
                            pointCx: float, pointCy: float) -> [float]:  # return angle
    # 公式參考自:
    # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
    vectorU = [pointAx - centerPointX, pointAy - centerPointY]  # 向量U
    vectorV = [pointCx - centerPointX, pointCy - centerPointY]  # 向量V
    lengthOfVectorU = math.sqrt((vectorU[0] * vectorU[0] + vectorU[1] * vectorU[1]))
    lengthOfVectorV = math.sqrt((vectorV[0] * vectorV[0] + vectorV[1] * vectorV[1]))
    innerProduct_of_UV = vectorU[0] * vectorV[0] + vectorU[1] * vectorV[1]  # 向量U 與 向量V的內積
    cosTheta = innerProduct_of_UV / (lengthOfVectorU * lengthOfVectorV)  # 向量U 與 向量V夾角的cosine值
    angle_in_degree = math.acos(cosTheta) * 180 / math.pi  # 弧度*180/pi轉成角度
    return angle_in_degree

def judgehand1FingersBendStatus():
    # ======below is to judge if fingers of hand1 has bent======
    # Finished :focus on thumb bend accuracy and Three judge accuracy(use angle to judge if finger has bent)
    # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
    try:
        hand1_thumbAngle = calculate_3_point_angle(hand1_coordinates[4][0], hand1_coordinates[4][1],
                                                   hand1_coordinates[3][0], hand1_coordinates[3][1],
                                                   hand1_coordinates[2][0], hand1_coordinates[2][1])
        hand1_indexFingerAngle = calculate_3_point_angle(hand1_coordinates[5][0], hand1_coordinates[5][1],
                                                         hand1_coordinates[6][0], hand1_coordinates[6][1],
                                                         hand1_coordinates[7][0], hand1_coordinates[7][1])
        hand1_middleFingerAngle = calculate_3_point_angle(hand1_coordinates[9][0], hand1_coordinates[9][1],
                                                          hand1_coordinates[10][0], hand1_coordinates[10][1],
                                                          hand1_coordinates[11][0], hand1_coordinates[11][1])
        hand1_ringFingerAngle = calculate_3_point_angle(hand1_coordinates[13][0], hand1_coordinates[13][1],
                                                        hand1_coordinates[14][0], hand1_coordinates[14][1],
                                                        hand1_coordinates[15][0], hand1_coordinates[15][1])
        hand1_pinkyAngle = calculate_3_point_angle(hand1_coordinates[17][0], hand1_coordinates[17][1],
                                                   hand1_coordinates[18][0], hand1_coordinates[18][1],
                                                   hand1_coordinates[19][0], hand1_coordinates[19][1])
    except:
        print("Oops found Missing Joints")

    if hand1_thumbAngle < 150:
        hand1_fingerBendStatus[0] = 1
        # cv2.putText(image, "thumb bent, angle:" + str(hand1_thumbAngle), (30, 30), cv2.FONT_HERSHEY_COMPLEX,
        #             0.6, (255, 255, 255), 2)
    else:
        hand1_fingerBendStatus[0] = 0
    if hand1_indexFingerAngle < 135:
        hand1_fingerBendStatus[1] = 1
        # cv2.putText(image, "index finger bent, angle:" + str(hand1_indexFingerAngle), (30, 60),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand1_fingerBendStatus[1] = 0
    if hand1_middleFingerAngle < 135:
        hand1_fingerBendStatus[2] = 1
        # cv2.putText(image, "middle finger bent, angle:" + str(hand1_middleFingerAngle), (30, 90),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand1_fingerBendStatus[2] = 0
    if hand1_ringFingerAngle < 135:
        hand1_fingerBendStatus[3] = 1
        # cv2.putText(image, "ring finger bent, angle:" + str(hand1_ringFingerAngle), (30, 120),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand1_fingerBendStatus[3] = 0
    if hand1_pinkyAngle < 135:
        hand1_fingerBendStatus[4] = 1
        # cv2.putText(image, "pinky bent, angle:" + str(hand1_pinkyAngle), (30, 150), cv2.FONT_HERSHEY_COMPLEX,
        #             0.6, (255, 255, 255), 2)
    else:
        hand1_fingerBendStatus[4] = 0
    # ======above is to judge if fingers of hand1 has bent======


def judgehand2FingersBendStatus():
    # ======below is to judge if fingers of hand2 has bent======
    # Finished :focus on thumb bend accuracy and Three judge accuracy(use angle to judge if finger has bent)
    # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
    try:
        hand2_thumbAngle = calculate_3_point_angle(hand2_coordinates[4][0], hand2_coordinates[4][1],
                                                   hand2_coordinates[3][0], hand2_coordinates[3][1],
                                                   hand2_coordinates[2][0], hand2_coordinates[2][1])
        hand2_indexFingerAngle = calculate_3_point_angle(hand2_coordinates[5][0], hand2_coordinates[5][1],
                                                         hand2_coordinates[6][0], hand2_coordinates[6][1],
                                                         hand2_coordinates[7][0], hand2_coordinates[7][1])
        hand2_middleFingerAngle = calculate_3_point_angle(hand2_coordinates[9][0], hand2_coordinates[9][1],
                                                          hand2_coordinates[10][0], hand2_coordinates[10][1],
                                                          hand2_coordinates[11][0], hand2_coordinates[11][1])
        hand2_ringFingerAngle = calculate_3_point_angle(hand2_coordinates[13][0], hand2_coordinates[13][1],
                                                        hand2_coordinates[14][0], hand2_coordinates[14][1],
                                                        hand2_coordinates[15][0], hand2_coordinates[15][1])
        hand2_pinkyAngle = calculate_3_point_angle(hand2_coordinates[17][0], hand2_coordinates[17][1],
                                                   hand2_coordinates[18][0], hand2_coordinates[18][1],
                                                   hand2_coordinates[19][0], hand2_coordinates[19][1])
    except:
        print("Oops found Missing Joints")

    if hand2_thumbAngle < 150:
        hand2_fingerBendStatus[0] = 1
        # cv2.putText(image, "thumb bent, angle:" + str(hand2_thumbAngle), (30, 30), cv2.FONT_HERSHEY_COMPLEX,
        #             0.6, (255, 255, 255), 2)
    else:
        hand2_fingerBendStatus[0] = 0
    if hand2_indexFingerAngle < 135:
        hand2_fingerBendStatus[1] = 1
        # cv2.putText(image, "index finger bent, angle:" + str(hand2_indexFingerAngle), (30, 60),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand2_fingerBendStatus[1] = 0
    if hand2_middleFingerAngle < 135:
        hand2_fingerBendStatus[2] = 1
        # cv2.putText(image, "middle finger bent, angle:" + str(hand2_middleFingerAngle), (30, 90),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand2_fingerBendStatus[2] = 0
    if hand2_ringFingerAngle < 135:
        hand2_fingerBendStatus[3] = 1
        # cv2.putText(image, "ring finger bent, angle:" + str(hand2_ringFingerAngle), (30, 120),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    else:
        hand2_fingerBendStatus[3] = 0
    if hand2_pinkyAngle < 135:
        hand2_fingerBendStatus[4] = 1
        # cv2.putText(image, "pinky bent, angle:" + str(hand2_pinkyAngle), (30, 150), cv2.FONT_HERSHEY_COMPLEX,
        #             0.6, (255, 255, 255), 2)
    else:
        hand2_fingerBendStatus[4] = 0
    # ======above is to judge if fingers of hand2 has bent======


def hand1GestureJudge():
    # ======below is hand1 gesture judge======
    if hand1_fingerBendStatus == [1, 1, 1, 1, 1]:
        hand1GestureJudgeResult = "Zero"
    elif hand1_fingerBendStatus == [1, 0, 1, 1, 1]:
        hand1GestureJudgeResult = "One"
    elif hand1_fingerBendStatus == [1, 0, 0, 1, 1]:
        hand1GestureJudgeResult = "Two"
    elif hand1_fingerBendStatus == [1, 0, 0, 0, 1]:
        hand1GestureJudgeResult = "Three"
    elif hand1_fingerBendStatus == [1, 0, 0, 0, 0]:
        hand1GestureJudgeResult = "Four"
    elif hand1_fingerBendStatus == [0, 0, 0, 0, 0]:
        hand1GestureJudgeResult = "Five"
    elif hand1_fingerBendStatus == [0, 1, 1, 1, 0]:
        hand1GestureJudgeResult = "Six"
    elif hand1_fingerBendStatus == [0, 0, 1, 1, 1]:
        hand1GestureJudgeResult = "Seven"
    elif hand1_fingerBendStatus == [0, 0, 0, 1, 1]:
        hand1GestureJudgeResult = "Eight"
    elif hand1_fingerBendStatus == [0, 0, 0, 0, 1]:
        hand1GestureJudgeResult = "Nine"
    elif hand1_fingerBendStatus == [1, 1, 0, 0, 0]:
        hand1GestureJudgeResult = "OK"
    elif hand1_fingerBendStatus == [0, 0, 1, 1, 0]:
        hand1GestureJudgeResult = "SpiderMan"
    elif hand1_fingerBendStatus == [1, 0, 1, 1, 0]:
        hand1GestureJudgeResult = "Rock"
    else:
        hand1GestureJudgeResult = "Undefined"
    cv2.putText(image, hand1_label + hand1GestureJudgeResult, (int(cam_width / 3 - 60), 60), cv2.FONT_HERSHEY_COMPLEX,
                1.5, (0, 255, 0), 2)
    # ======above is hand1 gesture judge======
    

def hand2GestureJudge():
    # ======below is hand2 gesture judge======
    if hand2_fingerBendStatus == [1, 1, 1, 1, 1]:
        hand2GestureJudgeResult = "Zero"
    elif hand2_fingerBendStatus == [1, 0, 1, 1, 1]:
        hand2GestureJudgeResult = "One"
    elif hand2_fingerBendStatus == [1, 0, 0, 1, 1]:
        hand2GestureJudgeResult = "Two"
    elif hand2_fingerBendStatus == [1, 0, 0, 0, 1]:
        hand2GestureJudgeResult = "Three"
    elif hand2_fingerBendStatus == [1, 0, 0, 0, 0]:
        hand2GestureJudgeResult = "Four"
    elif hand2_fingerBendStatus == [0, 0, 0, 0, 0]:
        hand2GestureJudgeResult = "Five"
    elif hand2_fingerBendStatus == [0, 1, 1, 1, 0]:
        hand2GestureJudgeResult = "Six"
    elif hand2_fingerBendStatus == [0, 0, 1, 1, 1]:
        hand2GestureJudgeResult = "Seven"
    elif hand2_fingerBendStatus == [0, 0, 0, 1, 1]:
        hand2GestureJudgeResult = "Eight"
    elif hand2_fingerBendStatus == [0, 0, 0, 0, 1]:
        hand2GestureJudgeResult = "Nine"
    elif hand2_fingerBendStatus == [1, 1, 0, 0, 0]:
        hand2GestureJudgeResult = "OK"
    elif hand2_fingerBendStatus == [0, 0, 1, 1, 0]:
        hand2GestureJudgeResult = "SpiderMan"
    elif hand2_fingerBendStatus == [1, 0, 1, 1, 0]:
        hand2GestureJudgeResult = "Rock"
    else:
        hand2GestureJudgeResult = "Undefined"
    cv2.putText(image, hand2_label + hand2GestureJudgeResult, (int(cam_width / 3 - 60), 120), cv2.FONT_HERSHEY_COMPLEX,
                1.5, (0, 255, 0), 2)
    # ======above is hand2 gesture judge======3


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
        hand1_coordinates = []
        for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_px = normalized_3_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
            if landmark_px:
                hand1_coordinates.append(landmark_px)

        hand1_coordinates = np.array(hand1_coordinates)

        hand1_label = results.multi_handedness[0].classification[0].label
        # Extract classification label from results.multi_handedness

        if len(results.multi_hand_landmarks) == 2:
            hand2_coordinates = []
            for idx, landmark in enumerate(results.multi_hand_landmarks[1].landmark):
                if landmark.visibility < 0 or landmark.presence < 0:
                    continue
                landmark_px = normalized_3_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
                if landmark_px:
                    hand2_coordinates.append(landmark_px)

            hand2_coordinates = np.array(hand2_coordinates)

            hand2_label = results.multi_handedness[1].classification[0].label
            # Extract classification label from results.multi_handedness

            # TODO: judge hand2 finger bend status
            # TODO: hand2 gestrue judge

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
