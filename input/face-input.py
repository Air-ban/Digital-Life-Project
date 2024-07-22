import cv2
import os
import numpy as np
import sys
weight_num = 0.4 #提权系数

# 加载Haar级联文件
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# SIFT 特征检测器
sift = cv2.SIFT_create()

#创建训练集文件夹
folder_path = 'dataset'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if os.listdir(folder_path):
    print("已检测到基础面部数据")
else:
    print("未检测到基础面部数据，请添加后再试")
    sys.exit()

# 数据集路径，替换本地数据集文件
dataset_path = 'dataset'

# 存储已知人脸的特征点
known_faces_features = []
known_names = []

with open('face_recognition_results.txt', 'w') as file:
        file.truncate(0)  # 清空文件内容

# 加载已知人脸并提取特征点
for name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, name)
    if os.path.isdir(person_dir):
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = img[y:y+h, x:x+w]
                keypoints, descriptors = sift.detectAndCompute(face_roi, None)
                known_faces_features.append(descriptors)
                known_names.append(name)

# 初始化摄像头
video_capture = cv2.VideoCapture(0)

# 创建FLANN匹配器
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    # 从摄像头读取一帧
    ret, frame = video_capture.read()
    if not ret:
        print("无法从摄像头读取帧")
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测帧中的人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_roi = gray[y:y+h, x:x+w]

        # 检测特征点
        keypoints, descriptors = sift.detectAndCompute(face_roi, None)

        # 进行特征点匹配
        matches = []
        if descriptors is not None:
            for i, known_descriptors in enumerate(known_faces_features):
                matches.append(flann.knnMatch(descriptors, known_descriptors, k=2))

        # 确定最佳匹配
        best_match_name = "unknow"
        best_match_score = 0
        for i, match in enumerate(matches):
            if len(match) > 0:
                # 计算好的匹配数量
                good_matches = sum(1 for m, n in match if m.distance < weight_num * n.distance)
                # 更新最佳匹配
                if good_matches > best_match_score:
                    best_match_score = good_matches
                    best_match_name = known_names[i]

        # 在视频帧上绘制边框和名称
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, best_match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        with open('face_recognition_results.txt', 'a') as file:
            file.write(f"{best_match_name}\n")

    # 显示结果
    cv2.imshow('Video', frame)


    # 按 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
video_capture.release()
cv2.destroyAllWindows()
