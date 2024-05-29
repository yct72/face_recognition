# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm, tree
import os
from utils import *

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
known_face_encodings = dict()
names = []
tolerance = 0.35

# Training directory
train_dir = os.listdir('./train_dir/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("./train_dir/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        print(person_img)
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("./train_dir/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
            if person not in known_face_encodings:
                known_face_encodings[person] = [face_enc]
            else:
                known_face_encodings[person].append(face_enc)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# Decision Tree Classifier
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(encodings, names)

# Predict all the faces in the test image using the trained classifier
test_dir = os.listdir('./test_dir/')
for test_image in test_dir:
    if (str(test_image) == "output"):
        continue
    print("For image: ", test_image)
    # Load the test image with unknown faces into a numpy array
    test_image_file = str(test_image)
    test_image = face_recognition.load_image_file("./test_dir/" + test_image)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
    print("Found:")
    firsttime = True
    for i in range(no):
        top, right, bottom, left = face_locations[i]
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        # probabilities = clf.predict_proba([test_image_enc])[0]
        # predict_idx = list(clf.classes_).index(name)
        # confidence = probabilities[predict_idx]
        # print(f"name: {names}, probabilities: {probabilities}, confidence: {confidence}")
        distances = face_recognition.face_distance(known_face_encodings[str(name[0])], test_image_enc)
        result = list(distances <= tolerance)
        print("--------------------")
        if True in result:
            print(*name, f"right bound: {right}")
            label_face('./test_dir/', test_image_file, face_locations[i], str(name[0]), firsttime)
        else:
            print("unknown", f"right bound: {right}")
            label_face('./test_dir/', test_image_file, face_locations[i], "unknown", firsttime)
        # scores = clf.decision_function([test_image_enc])
        # for class_label, score in zip(clf.classes_, scores[0]):
            # print(f"Class: {class_label}, Score: {score:.2f}")
        firsttime = False
