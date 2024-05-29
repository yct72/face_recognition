import cv2
import numpy as np

name2id = {
    "yohsin": "1",
    "意欣": "2",
    "政雅": "3",
    "翊綺": "4",
    "unknown": "unknown",
}

def label_face(directory, file, face_location, name, firsttime):
    new_file = directory + 'output/' + file.split(".")[0] + '_label.jpg'
    if firsttime:
        img = cv2.imread(directory + file)
    else: 
        img = cv2.imread(new_file)

    top, right, bottom, left = face_location
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_np = np.array(gray,'uint8')              

    # draw bounding box
    cv2.rectangle(img,(left,bottom),(right,top),(0,255,0),2)           
    cv2.putText(img, name2id[name], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # write to file
    cv2.imwrite(new_file, img)