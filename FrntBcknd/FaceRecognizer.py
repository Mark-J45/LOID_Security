import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('Haar_Cascade.xml')
dir = r'C:\Users\dijon\Downloads\FaceRecog\Photos'

Face = []
Name = []
FR = cv.face.LBPHFaceRecognizer_create()
people = []
label_person = 0

def load_model():
    Face = np.load('Faces.npy',allow_pickle=True)
    Name = np.load('Names.npy',allow_pickle=True) 
    FR.read('TrainedModel.yml')

def rescale(frame,scale=0.75):
        width = int(frame.shape[1] *scale)
        height = int(frame.shape[0] *scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def check_face(frame):
    frame = cv.resize(frame, (680, 660))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 10)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + h]
        centerx = int(x + (w / 2))
        centery = int(y + (h / 2))

        label_person, confidence = FR.predict(faces_roi)
        cv.putText(frame, str(people[label_person]), (x + 5, y + 25), cv.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 0, 0),
                   thickness=1)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x + 1, y + 1), (x + w - 1, centery), (255, 0, 0), thickness=1)

        # Check if the label matches a specific condition
        

    return frame


def main():


    load_model()

    for i in os.listdir(dir):
        people.append(i) 

    print(people)

    capture = cv.VideoCapture(0)

    while True:
        isTrue, f  = capture.read()
        f = cv.flip(f,1)
        frame = check_face(f)
        cv.imshow("Live Video",frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    
main()