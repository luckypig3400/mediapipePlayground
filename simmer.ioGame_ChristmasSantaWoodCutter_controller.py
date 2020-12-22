# for this game on i.simmer.io:Christmas Santa Wood Cutter
# https://simmer.io/@Zathos/christmas-santa-wood-cutter
# 左右手握拳，對應遊戲上砍左邊/砍右邊 偵測到握拳自動點擊遊戲上對應的控制區一下

from multiprocessing import Process

def twoHandsGestureDetector():
    twoHandGesture
    pass  # pass 是空的語句 我覺得可以作為 } 使用 ，讓自己可以更方便的讀python程式碼

def watcher():

    if (twoHandGesture.hand1_label == "Left" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
            twoHandGesture.hand2_label == "Left" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
        print("YA~ should send left click signal on Left side of screen")

    if (twoHandGesture.hand1_label == "Right" and twoHandGesture.hand1GestureJudgeResult == "Zero") or (
            twoHandGesture.hand2_label == "Right" and twoHandGesture.hand2GestureJudgeResult == "Zero"):
        print("YA~ should send left click signal on Right side of screen")

    # TODO: fix no reaction to the imported variables
    pass


if __name__ == '__main__':
    # https://stackoverflow.com/questions/7207309/how-to-run-functions-in-parallel
    p1 = Process(target=watcher)
    p1.start()
    p2 = Process(target=twoHandsGestureDetector)
    p2.start()
    p1.join()
    p2.join()
    pass
