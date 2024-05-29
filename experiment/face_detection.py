# Find faces in an image

"""
Structure:
        <test_dir>/
            <people_1>.jpg
            <people_2>.jpg
            .
            .
            <people_n>.jpg
"""

import face_recognition
from PIL import Image
import os
import cv2
import numpy as np


# Testing directory
test_dir = os.listdir('./test_dir/')

# cv
detector = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recognizer = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

# Loop through each person in the testing directory
for person_img in test_dir:

    # Loop through each testing image for the current person
    print(person_img)
    # Get the face encodings for the face in each image file
    image = face_recognition.load_image_file("./test_dir/"  + person_img)
    face_locations = face_recognition.face_locations(image)
    cont = False
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        if cont:
            img = cv2.imread("output.jpg")        
        else:
            img = cv2.imread("./test_dir/"  + person_img)           # 開啟照片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        img_np = np.array(gray,'uint8')               # 轉換成指定編碼的 numpy 陣列

        faces.append(img_np[bottom:top,left:right])         # 記錄張忠謀人臉的位置和大小內像素的數值
        ids.append(1)                             # 記錄張忠謀人臉對應的id為1(只能是整數)
        cv2.rectangle(img,(left,bottom),(right,top),(0,255,0),2)            # 標記人臉外框
        cv2.putText(img, "hi", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # cv2.imshow('recognizer_train', img)
        # cv2.waitKey(0)
        # 顯示框出來的圖片(jupyter會直接顯示在block下方)
        # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # write to file
        cv2.imwrite('output.jpg', img)
        cont = True
        # plt.imshow(image_rgb)
        # plt.show()

           