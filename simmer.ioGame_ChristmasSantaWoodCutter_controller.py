# for this game on i.simmer.io:Christmas Santa Wood Cutter
# https://simmer.io/@Zathos/christmas-santa-wood-cutter
# 左右手握拳，對應遊戲上砍左邊/砍右邊 偵測到握拳自動點擊遊戲上對應的控制區一下
import os

import MyVersion_HandGestureDetectorUsingAngle_support2hands as twoHandGesture

# os.system("MyVersion_HandGestureDetectorUsingAngle_support2hands.py")

if (twoHandGesture.hand1_label == "Left" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
        twoHandGesture.hand2_label == "Left" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
    print("YA~ should send left click signal on Left side of screen")

if (twoHandGesture.hand1_label == "Right" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
        twoHandGesture.hand2_label == "Right" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
    print("YA~ should send left click signal on Right side of screen")

# TODO: fix no reaction to the imported variables
