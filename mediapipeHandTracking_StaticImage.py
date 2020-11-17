import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)
for idx, file in enumerate(file_list):
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  image = cv2.flip(cv2.imread(file), 1)
  # Convert the BGR image to RGB before processing.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print handedness and draw hand landmarks on the image.
  print('handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    continue
  annotated_image = image.copy()
  for hand_landmarks in results.multi_hand_landmarks:
    print('hand_landmarks:', hand_landmarks)
    mp_drawing.draw_landmarks(
        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
  cv2.imwrite(
      '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1))
hands.close()
