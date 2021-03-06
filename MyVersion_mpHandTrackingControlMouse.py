import cv2
import mediapipe as mp
import pyautogui
import time

webcam_id = 1
window_name = 'Hand Tracking Control Mouse Example'
desktop_width, desktop_height = pyautogui.size()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mouseXmarginFix = 30  # 滑鼠x座標調整，會依照手指所在區域對應不同調整辦法
mouseYmarginFix = 30  # 滑鼠y座標調整，會依照手指所在區域對應不同調整辦法
userMousePositionZoomFactor = 1.06  # 除了乘上螢幕解析度外，額外乘以的係數
lastMouseMoveMillis = 0
lastMouseLeftClickMillis = 0
lastMouseRightClickMillis = 0

pyautogui.PAUSE = 0  # 讓pyautogui在使用移動函式後不要暫停系統

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
while cap.isOpened():
    millis = int(round(time.time() * 1000))
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
                if idx == 4:  # thumb TIP x,y position
                    thumbTIP_x, thumbTIP_y = idx_to_coordinates[4]

                    # debug coordinate info for thumb
                    cv2.putText(image, "(" + str(thumbTIP_x) + "," + str(thumbTIP_y) + ")", (thumbTIP_x, thumbTIP_y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (255, 255, 255), 2)

                if idx == 5:  # index Finger MCP x,y position
                    indexFingerMCP_x, indexFingerMCP_y = idx_to_coordinates[5]
                    # if idx_to_coordinates[5] is not None then store them in x,y variables

                    # debug coordinate info for index finger
                    cv2.putText(image, "(" + str(indexFingerMCP_x) + "," + str(indexFingerMCP_y) + ")",
                                (indexFingerMCP_x, indexFingerMCP_y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (255, 255, 255), 2)

                if idx == 8:  # index Finger TIP x,y position
                    indexFingerTIP_x, indexFingerTIP_y = idx_to_coordinates[8]
                    # if idx_to_coordinates[8] is not None then store them in x,y variables

                    # debug coordinate info for index finger
                    cv2.putText(image, "(" + str(indexFingerTIP_x) + "," + str(indexFingerTIP_y) + ")",
                                (indexFingerTIP_x, indexFingerTIP_y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (255, 255, 255), 2)

                if idx == 12:  # middle Finger TIP x,y position
                    middleFingerTIP_x, middleFingerTIP_y = idx_to_coordinates[12]

                    # debug coordinate info for middle finger
                    cv2.putText(image, "(" + str(middleFingerTIP_x) + "," + str(middleFingerTIP_y) + ")",
                                (middleFingerTIP_x, middleFingerTIP_y),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                (255, 255, 255), 2)

        # Screen Monitor
        window_x, window_y, window_w, window_h = cv2.getWindowImageRect(window_name)
        # TODO divide webcam area to four grid, each grid will use different Mathematical formula to calculate mouse position
        if abs(lastMouseMoveMillis - millis) >= 30:  # 每30毫秒移動一次滑鼠
            # 使用食指的MCP關節點來移動鼠標位置可以大幅減少手指作出點擊動作時造成的抖動
            pyautogui.moveTo(indexFingerMCP_x / window_w * desktop_width, indexFingerMCP_y / window_h * desktop_height)
            lastMouseMoveMillis = millis
        if abs(thumbTIP_x - indexFingerTIP_x) <= 18 and abs(thumbTIP_y - indexFingerTIP_y) <= 18:
            if abs(lastMouseLeftClickMillis - millis) >= 900:  # 每900毫秒可以點擊滑鼠左鍵
                pyautogui.leftClick()
                lastMouseLeftClickMillis = millis
        if abs(thumbTIP_x - middleFingerTIP_x) <= 18 and abs(thumbTIP_y - middleFingerTIP_y) <= 18:
            if abs(lastMouseRightClickMillis - millis) >= 900:  # 每900毫秒可以點擊滑鼠右鍵
                pyautogui.rightClick()
                lastMouseRightClickMillis = millis

    cv2.imshow(window_name, image)
    if cv2.waitKey(5) & 0xFF == 27:
        print(idx_to_coordinates)
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

# INDEX_FINGER_DIP = <HandLandmark.INDEX_FINGER_DIP: 7>
# INDEX_FINGER_MCP = <HandLandmark.INDEX_FINGER_MCP: 5>
# INDEX_FINGER_PIP = <HandLandmark.INDEX_FINGER_PIP: 6>
# INDEX_FINGER_TIP = <HandLandmark.INDEX_FINGER_TIP: 8>
#
# MIDDLE_FINGER_DIP = <HandLandmark.MIDDLE_FINGER_DIP: 11>
# MIDDLE_FINGER_MCP = <HandLandmark.MIDDLE_FINGER_MCP: 9>
# MIDDLE_FINGER_PIP = <HandLandmark.MIDDLE_FINGER_PIP: 10>
# MIDDLE_FINGER_TIP = <HandLandmark.MIDDLE_FINGER_TIP: 12>
#
# PINKY_DIP = <HandLandmark.PINKY_DIP: 19>
# PINKY_MCP = <HandLandmark.PINKY_MCP: 17>
# PINKY_PIP = <HandLandmark.PINKY_PIP: 18>
# PINKY_TIP = <HandLandmark.PINKY_TIP: 20>
#
# RING_FINGER_DIP = <HandLandmark.RING_FINGER_DIP: 15>
# RING_FINGER_MCP = <HandLandmark.RING_FINGER_MCP: 13>
# RING_FINGER_PIP = <HandLandmark.RING_FINGER_PIP: 14>
# RING_FINGER_TIP = <HandLandmark.RING_FINGER_TIP: 16>
#
# THUMB_CMC = <HandLandmark.THUMB_CMC: 1>
# THUMB_IP = <HandLandmark.THUMB_IP: 3>
# THUMB_MCP = <HandLandmark.THUMB_MCP: 2>
# THUMB_TIP = <HandLandmark.THUMB_TIP: 4>
#
# WRIST = <HandLandmark.WRIST: 0>
