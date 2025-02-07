import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def get_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        points = [(int(p.x * w), int(p.y * h)) for p in face_landmarks]
        return points
    return []

def affine_transform(points1, points2):
    # indices = [1, 2, 3]  # 假设这些是关键点的索引
    # pts1 = np.float32([points1[i] for i in indices])
    # pts2 = np.float32([points2[i] for i in indices])
    # matrix = cv2.getAffineTransform(pts2, pts1)
    pts1 = np.float32(points1)
    pts2 = np.float32(points2)
    matrix, _ = cv2.estimateAffine2D(pts2, pts1)
    return matrix

def swap_faces(image1, image2):
    points1 = get_face_landmarks(image1)
    points2 = get_face_landmarks(image2)
    
    if points1 and points2:
        transform_matrix = affine_transform(points1, points2)
        transformed_image = cv2.warpAffine(image2, transform_matrix, (image1.shape[1], image1.shape[0]))
        
        mask = np.zeros_like(image1, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(points1, dtype=np.int32), (255, 255, 255))
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        

        return transformed_image
    return image2
# 读取图片
image1 = cv2.imread('meizi.jpg')
image2 = cv2.imread('mingxing.jpg')

# 换脸
result_image = swap_faces(image1, image2)

# 显示结果
cv2.imshow('Swapped Face', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()