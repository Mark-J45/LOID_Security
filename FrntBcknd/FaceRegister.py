from multiprocessing.pool import TERMINATE
import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('Haar_Cascade.xml')
dir = r'C:\newpc\LOID_Detection\Photos'
viddir = r'C:\Users\dijon\Downloads\videorecog\mark.mp4'

sname = input("INPUT NAME : ")



print(sname)

people = []
for i in os.listdir(dir):
    people.append(i)

print(people)

path = os.path.join(dir,sname)

if sname not in people:
    
    os.makedirs(path)

count = 0
for j in os.listdir(path):
   count = count+1
print(f"number of photos = " + str(count))
capture = cv.VideoCapture(0)

def rescale(frame,scale=0.75):
        width = int(frame.shape[1] *scale)
        height = int(frame.shape[0] *scale)
        dimensions = (width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


while True:
    isTrue, frame  = capture.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray,1.1,10)
    
    for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+h]
            if count <300:
                img = cv.imwrite(path + "\image" + str(count) + ".jpg",frame) 
                print("Saved new image #" + str(count))
                count = count +1
                percentage = count/3
                cv.putText(frame, str(int(percentage))+"%" ,(x+50,y+50), cv.FONT_HERSHEY_COMPLEX,fontScale=1.0, color =(0,255,0),thickness= 2)
            else:
                cv.putText(frame, str(sname) ,(x+50,y+50), cv.FONT_HERSHEY_COMPLEX,fontScale=1.0, color =(0,255,0),thickness= 2)
            
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
           

    cv.imshow("Live Video",frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
