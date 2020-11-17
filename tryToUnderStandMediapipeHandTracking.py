import cv2
import mediapipe as mp
import pyautogui
from mediapipe.framework.formats import landmark_pb2

webCamID = 1

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_landmarksOutputCount = 1
desktop_width, desktop_height = pyautogui.size()

myhand_landmarkList = []
myhand_landmarkDic = {}
# dictionary 用來包每個手部關鍵點的資訊
# list 用來儲存及標示每個手部關鍵點的字典分別代表再圖上代表哪個點

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

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.8, min_tracking_confidence=0.66)
cap = cv2.VideoCapture(webCamID)
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
            # print("第", hand_landmarksOutputCount, "次輸出landmark")
            # print("第", hand_landmarksOutputCount, "次輸出HAND_CONNECTIONS")
            # print(mp_hands.HAND_CONNECTIONS)
            # hand_landmarksOutputCount += 1

        for idx, landmark in enumerate(hand_landmarks.landmark[0:8]):
            print(idx, landmark)

        # mouseX = results.multi_hand_landmarks[8]['x'] * desktop_width
        # mouseY = results.multi_hand_landmarks[8]['y'] * desktop_height
        # print(mouseX , mouseY)

        # print("len :",len(results.multi_hand_landmarks))
        # print("type:", type(results.multi_hand_landmarks))

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()