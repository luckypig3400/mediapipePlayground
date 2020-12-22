# for this game on i.simmer.io:Christmas Santa Wood Cutter
# https://simmer.io/@Zathos/christmas-santa-wood-cutter
# 左右手握拳，對應遊戲上砍左邊/砍右邊 偵測到握拳自動點擊遊戲上對應的控制區一下

from multiprocessing import Process


def twoHandsGestureDetector():
    # import MyVersion_HandGestureDetectorUsingAngle_support2hands as twoHandGesture
    print("Hello")


def func1():
    print('func1: starting')
    for i in range(600000):
        print("func1 progress:" + str(i))
        pass
    print('func1: finishing')


def func2():
    print('func2: starting')
    for i in range(600000):
        print("func2 progress:" + str(i))
        pass
    print('func2: finishing')


if __name__ == '__main__':
    # https://stackoverflow.com/questions/7207309/how-to-run-functions-in-parallel
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p1.join()
    p2.join()

"""
print(twoHandGesture.hand1_label)
print(twoHandGesture.hand2_label)

if (twoHandGesture.hand1_label == "Left" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
        twoHandGesture.hand2_label == "Left" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
    print("YA~ should send left click signal on Left side of screen")

if (twoHandGesture.hand1_label == "Right" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
        twoHandGesture.hand2_label == "Right" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
    print("YA~ should send left click signal on Right side of screen")

# TODO: fix no reaction to the imported variables

"""
