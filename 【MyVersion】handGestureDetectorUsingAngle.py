import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

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
hand1_fingerBendStatus = [0, 0, 0, 0, 0]  # 0~4 : thumb~pinky

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

        #  ====== fetch Left/Right label from results.multi_handedness ======  #
        print('Handedness:', results.multi_handedness)  # above mediapipe 0.8.1 provide handedness to show Left or Right hand

        print(results.multi_handedness[0])
        """
        data inside results.multi_handedness[0]:
        classification {
            index: 0
            score: 0.999981701374054
            label: "Left"
        }
        """

        print(type(results.multi_handedness))  # <class 'list'>
        print(type(
            results.multi_handedness[0]))  # <class 'mediapipe.framework.formats.classification_pb2.ClassificationList'>

        print(results.multi_handedness[0].classification[
                  0].label)  # Extract classification label from results.multi_handedness

        print(len(results.multi_hand_landmarks))  # 輸出偵測到幾隻手
        #  ====== fetch Left/Right label from results.multi_handedness ======  #

        image_rows, image_cols, _ = image.shape
        hand1_coordinates = []
        for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_px = normalized_3_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
            if landmark_px:
                hand1_coordinates.append(landmark_px)

        hand2_coordinates = []
        if len(results.multi_hand_landmarks) == 2:
            for idx, landmark in enumerate(results.multi_hand_landmarks[1].landmark):
                if landmark.visibility < 0 or landmark.presence < 0:
                    continue
                landmark_px = normalized_3_pixel_coordinates(landmark.x, landmark.y, landmark.z, image_cols, image_rows)
                if landmark_px:
                    hand2_coordinates.append(landmark_px)

        hand1_coordinates = np.array(hand1_coordinates)

        # below is to judge if finger has bent
        # Finished :focus on thumb bend accuracy and Three judge accuracy(use angle to judge if finger has bent)
        # 已知三點座標求夾角:https://tw.answers.yahoo.com/question/index?qid=20081223000016KK00623
        try:
            try:
                hand1_thumbAngle = calculate_3_point_angle(hand1_coordinates[4][0], hand1_coordinates[4][1],
                                                           hand1_coordinates[3][0], hand1_coordinates[3][1],
                                                           hand1_coordinates[2][0], hand1_coordinates[2][1])
                # print("hand1_hand1_thumbAngle:" + str(hand1_hand1_thumbAngle))
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
                # hand1_thumbAngle = 180
                print("Oops found missing thumb point")

            if hand1_thumbAngle < 150:
                hand1_fingerBendStatus[0] = 1
                # cv2.putText(image, "thumb bent, angle:" + str(hand1_thumbAngle), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                #             (255, 255, 255), 2)
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
                # cv2.putText(image, "pinky bent, angle:" + str(hand1_pinkyAngle), (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                #             (255, 255, 255), 2)
            else:
                hand1_fingerBendStatus[4] = 0
        except:
            print("Oops found Missing Joints")
        # above is to judge if finger has bent

        # below is hand gesture judge
        if hand1_fingerBendStatus == [1, 1, 1, 1, 1]:
            cv2.putText(image, "Zero", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [1, 0, 1, 1, 1]:
            cv2.putText(image, "One", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [1, 0, 0, 1, 1]:
            cv2.putText(image, "Two", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [1, 0, 0, 0, 1]:
            cv2.putText(image, "Three", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [1, 0, 0, 0, 0]:
            cv2.putText(image, "Four", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 0, 0, 0, 0]:
            cv2.putText(image, "Five", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 1, 1, 1, 0]:
            cv2.putText(image, "Six", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 0, 1, 1, 1]:
            cv2.putText(image, "Seven", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 0, 0, 1, 1]:
            cv2.putText(image, "Eight", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 0, 0, 0, 1]:
            cv2.putText(image, "Nine", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [1, 1, 0, 0, 0]:
            cv2.putText(image, "OK", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        elif hand1_fingerBendStatus == [0, 0, 1, 1, 0]:
            cv2.putText(image, "SpiderMan", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0),
                        2)
        elif hand1_fingerBendStatus == [1, 0, 1, 1, 0]:
            cv2.putText(image, "Rock", (int(cam_width / 2 - 60), 60), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
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
