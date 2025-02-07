import numpy as np
from PIL import ImageGrab
import cv2

# 设置录制参数
SCREEN_SIZE = (1920, 1080)
FILENAME = 'recorded_video.mp4'
FPS = 30.0
# 开始录制
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
out = cv2.VideoWriter(FILENAME, fourcc, FPS, SCREEN_SIZE)


while True:
    # 获取屏幕截图
    # img = pyautogui.screenshot()
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    print('recordin..')
    # 转换为OpenCV格式
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # 写入视频文件
    out.write(frame)
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame', 1920, 1080)

    # 检测按键
    if cv2.waitKey(1) == ord('q'):
        break



# 停止录制
out.release()
cv2.destroyAllWindows()

