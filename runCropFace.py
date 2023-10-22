
import cv2
import os 

def cropImage(img_rec):

    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml') # pylint: disable=no-member

    imagem  = cv2.imread('output/rec/' + img_rec)    
    gray    = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member
    faces   = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
    
    crop_img = ''
    for (x, y, w, h) in faces: 
                                
        img_face = 'output/rec/' + img_rec
                            
        crop_img = imagem[y:y+h, x:x+w]
        
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    
    return crop_img

if __name__ == '__main__': 
    
    path = 'E:\\PythonProjects\\tutorialSib23'
    
    cropImage('photo0.png')