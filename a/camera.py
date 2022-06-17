import cv2
import numpy as np

from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

with open('obj.names', 'r') as f:
        classes = f.read().splitlines()

model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')                

cap=cv2.VideoCapture(0)

labels={0:'Helmet',1:'Vest',2:'No Helmet',3:'No Vest'}
color={0:(0,255,0),1:(0,255,0),3:(255,0,0),4:(255,0,0)}

while(True):

    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    safety=model.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in safety :
    
        safety_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(safety_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),color[label],4)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color[label],4)
        cv2.putText(frame, labels[label], (x, y-10),cv2.FONT_ITALIC, 1,(255,255,255),4)
        
        if labels[label] =='No Helmet':
           sound.play()
           print("Beep")
        elif labels[label] == 'NO MASK':
                sound.play()
                print("Beep")


    cv2.imshow('Real Time Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()