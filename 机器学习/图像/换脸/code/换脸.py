#%%
import requests
import json
import simplejson
import base64
#  https://www.cnblogs.com/ddzzhh/p/17969911

#第一步：获取人脸关键点
def find_face(imgpath):
    """
    :param imgpath: 图片的地址
    :return: 一个字典类型的人脸关键点 如：{'top': 156, 'left': 108, 'width': 184, 'height': 184}
    """
    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect' #获取人脸信息的接口
    data = {
    "api_key":"x2NyKaa6vYuArYwat4x0-NpIbM9CrwGU",#访问url所需要的参数
    "api_secret":"OuHx-Xaey1QrORwdG7QetGG5JhOIC8g7",#访问url所需要的参数
    "image_url":imgpath, #图片地址
    "return_landmark":1
    }
    
    
    files = {'image_file':open(imgpath,'rb')} #定义一个字典存放图片的地址
    response = requests.post(http_url,data=data,files=files)
    res_con1 = response.content.decode('utf-8')
    res_json = simplejson.loads(res_con1)
    faces = res_json['faces']
    list = faces[0]
    rectangle = list['face_rectangle']
    # {'top': 122, 'left': 84, 'width': 123, 'height': 123}
    return rectangle
 
#第二步：实现换脸
def merge_face(image_url1,image_url2,image_url,number):
    """
    :param image_url1: 被换脸的图片路径
    :param image_url2: 换脸的图片路径
    :param image_url: 换脸后生成图片所保存的路径
    :param number: 换脸的相似度
    """
    #首先获取两张图片的人脸关键点
    face1 = find_face(image_url1)
    face2 = find_face(image_url2)
    #将人脸转换为字符串的格式
    rectangle1 = str(str(face1['top']) + "," + str(face1['left']) + "," + str(face1['width']) + "," + str(face1['height']))
    rectangle2 = str(str(face2['top']) + "," + str(face2['left']) + "," + str(face2['width']) + "," + str(face2['height']))
    #读取两张图片
    f1 = open(image_url1,'rb')
    f1_64 = base64.b64encode(f1.read())
    f1.close()
    f2 = open(image_url2, 'rb')
    f2_64 = base64.b64encode(f2.read())
    f2.close()
    
    url_add = 'https://api-cn.faceplusplus.com/imagepp/v1/mergeface' #实现换脸的接口
    data={
    "api_key": "x2NyKaa6vYuArYwat4x0-NpIbM9CrwGU",
    "api_secret": "OuHx-Xaey1QrORwdG7QetGG5JhOIC8g7",
    "template_base64":f1_64,
    "template_rectangle":rectangle1,
    "merge_base64":f2_64,
    "merge_rectangle":rectangle2,
    "merge_rate":number
    }
    response1 = requests.post(url_add,data=data)
    res_con1 = response1.content.decode('utf-8')
    res_dict = json.JSONDecoder().decode(res_con1)
    result = res_dict['result']
    imgdata = base64.b64decode(result)
    file=open(image_url,'wb')
    file.write(imgdata)
    file.close()
 
if __name__ == '__main__':
    face1="mingxing"
    face2="meizi"  
    face3=face1+"_"+face2 #把face2的脸换到face1图片中的脸上去
    image1 = r""+face1+".jpg"
    image2 = r""+face2+".jpg"
    image3 = r""+face3+".jpg"
    merge_face(image1,image2,image3,100)


#%%
import cv2
import numpy as np
import dlib
 
# 加载预训练的人脸检测模型
detector = dlib.get_frontal_face_detector()
 
# 读取源图像和目标图像
source_img_path = 'mingxing.jpg'
target_img_path = 'meizi.jpg'
source_image = cv2.imread(source_img_path, cv2.IMREAD_COLOR)
target_image = cv2.imread(target_img_path, cv2.IMREAD_COLOR)
 
# 检测源图像和目标图像中的人脸
detected_source = detector(source_image, 1)
detected_target = detector(target_image, 1)
 
# 假设只有一个人脸，获取第一个检测到的人脸区域
source_face = np.array([detected_source[0].left(), detected_source[0].top(),
                        detected_source[0].right(), detected_source[0].bottom()])
target_face = np.array([detected_target[0].left(), detected_target[0].top(),
                        detected_target[0].right(), detected_target[0].bottom()])
 
# 提取源图像的特征点（使用Dlib的68点标记预测）
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
source_landmarks = predictor(source_image, detected_source[0])
source_landmarks = np.array([(p.x, p.y) for p in source_landmarks.parts()])
 
# 创建一个矩阵，用于存储目标图像的特征点
target_landmarks = np.zeros((68, 2))
 
# 提取目标图像的特征点（这里需要根据实际情况调整）
# ...
 
# 根据特征点创建变换矩阵
tfm = cv2.getAffineTransform(source_landmarks, target_landmarks)
 
# 对目标图像进行变换
warped_target = cv2.warpAffine(target_image, tfm, (source_image.shape[1], source_image.shape[0]))
 
# 将变换后的图像的脸部与源图像的背景合并
warped_face = warped_target[source_face[1]:source_face[3], source_face[0]:source_face[2]]
source_image[source_face[1]:source_face[3], source_face[0]:source_face[2]] = warped_face
 
# 展示结果
cv2.imshow('Swapped Face', source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
image_face_fusion = pipeline(Tasks.image_face_fusion, 
                       model='damo/cv_unet-image-face-fusion_damo' )
template_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_template.jpg'
user_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facefusion_user.jpg'
result = image_face_fusion(dict(template=template_path, user=user_path) )

cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')

#%%
!python -m pip install modelscope -i   https://pypi.tuna.tsinghua.edu.cn/simple