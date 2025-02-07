#%%
import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 初始化 MediaPipe 绘图工具
mp_drawing = mp.solutions.drawing_utils

# 读取图像
image = cv2.imread('mingxing.jpg')

# 将图像从 BGR 转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行人脸关键点检测
results = face_mesh.process(image_rgb)

# 检查是否检测到人脸
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # 绘制人脸关键点
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

# 显示结果图像
cv2.imshow('Face Mesh', image)
cv2.waitKey(0)
cv2.destroyAllWindows()