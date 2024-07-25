import cv2
import os
import numpy as np
import sys

class face_input:
    def __init__(self):
        self.weight_num = 0.4 #提权系数
        # 加载Haar级联文件
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # SIFT 特征检测器
        self.sift = cv2.SIFT_create()
        #创建训练集文件夹
        self.folder_path = 'dataset'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        if os.listdir(self.folder_path):
            print("已检测到基础面部数据")
        else:
            print("未检测到基础面部数据，请添加后再试")
            sys.exit()
        # 数据集路径，替换本地数据集文件
        self.dataset_path = 'dataset'
        # 存储已知人脸的特征点
        self.known_faces_features = []
        self.name_list = []
        self.known_names = []
        # with open('face_recognition_results.txt', 'w') as file:
        #         file.truncate(0)  # 清空文件内容
        # 加载已知人脸并提取特征点
        for name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, name)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_file)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face_roi = img[y:y+h, x:x+w]
                        keypoints, descriptors = self.sift.detectAndCompute(face_roi, None)
                        self.known_faces_features.append(descriptors)
                        self.known_names.append(name)
        # 初始化摄像头
        self.video_capture = cv2.VideoCapture("0")
        # 创建FLANN匹配器
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
    def get_frame(self):# 从摄像头读取一帧
        ret, frame = self.video_capture.read()
        if not ret:
            #self.name_list.append('zzb')
            print("无法从摄像头读取帧")
            return None

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测帧中的人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = gray[y:y+h, x:x+w]

            # 检测特征点
            keypoints, descriptors = self.sift.detectAndCompute(face_roi, None)#这里有问题

            # 进行特征点匹配
            matches = []
            if descriptors is not None:
                for i, known_descriptors in enumerate(self.known_faces_features):
                    matches.append(self.flann.knnMatch(descriptors, known_descriptors, k=2))

            # 确定最佳匹配
            best_match_name = "unknow"
            best_match_score = 0
            for i, match in enumerate(matches):
                if len(match) > 0:
                    # 计算好的匹配数量
                    good_matches = sum(1 for m, n in match if m.distance < self.weight_num * n.distance)
                    # 更新最佳匹配
                    if good_matches > best_match_score:
                        best_match_score = good_matches
                        best_match_name = self.known_names[i]

            # 在视频帧上绘制边框和名称
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, best_match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #显示结果
            cv2.imshow('Video', frame)
            self.name_list.append(best_match_name)
    def proportion(self):
        # 检查 name_list 是否为空
        if len(self.name_list) == 0:
            print("名字列表为空，无法计算比例。")
            return None, None

        # 初始化一个字典，用于存储名字和对应的出现次数
        name_dict = {}
        for name in self.name_list:
            # 增加名字的计数
            name_dict[name] = name_dict.get(name, 0) + 1

        # 找出名字出现次数的最大值和最常见的名字列表
        max_count = max(name_dict.values())
        name = [name for name, count in name_dict.items() if count == max_count]
        name = name[0]
        # 计算出现频率最高的名字的比例
        total_count = len(self.name_list)
        proportions = [count / total_count for name, count in name_dict.items() if count == max_count]
        proportions=proportions[0]

        # 返回最常见的名字列表和它们的比例
        # 注意：如果存在多个最常见名字，这里会返回所有这些名字及其比例
        return name, proportions