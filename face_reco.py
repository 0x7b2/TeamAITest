#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import face_recognition

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
@ 人脸比较函数
@ return 0-100分数值
'''
def face_file_compare(file1, file2):
#    print(file1, file2)
    image1 = face_recognition.load_image_file(file1)
    m1facelist = face_recognition.face_encodings(image1)
    if len(m1facelist) == 0:
        return 0
    # 取第一张人脸
    m1_face_encoding = m1facelist[0]

    image2 = face_recognition.load_image_file(file2)
    m2facelist = face_recognition.face_encodings(image2)
    if len(m2facelist) == 0:
        return 0
    # 取第一张人脸
    m2_face_encoding = m2facelist[0]

    score = face_recognition.face_distance([m1_face_encoding], m2_face_encoding)
    return (1-score)*100

'''
@ 人脸位置
@ return (top, right, bottom, left)
'''
def face_file_locations(file):
    image = face_recognition.load_image_file(file)
    loc = face_recognition.face_locations(image)
    return loc

'''
@ 人脸特征点函数
@ 返回人脸数组
    facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]
'''
def face_file_landmarks(file):
    image = face_recognition.load_image_file(file)
    face_landmarks_list = face_recognition.face_landmarks(image)
    return face_landmarks_list
