
import dlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torchvision.transforms as transforms

from imutils import face_utils
from PIL import Image

from simple_3dviz import Mesh
from simple_3dviz.window import show
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames

import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

def createTxt(imagesp):
    
    subpastas = os.listdir(imagesp)
    
    for p in subpastas:

        pimg = imagesp + p
        
        if '.png' in pimg or '.jpg' in pimg:
        
            img = cv2.imread(pimg)
            print(img.shape)
            
            new = np.expand_dims(img, axis=0)     
            print(new.shape)
            
            
            # if len(img.shape) == 4:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            #     np.expand_dims(img, axis=0)
            #     cv2.imwrite(imagesp + p, img)
            #     print(img.shape) 
            # else:
                
            #     img = np.expand_dims(img, axis=0)
            #     print(img.shape)
            #     cv2.imwrite(imagesp + p, img) 
                
            #     print(img.shape) 
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            if len(rects) > 0:
        
                for rect in rects:

                    x = rect.left()
                    y = rect.top()
                    w = rect.right()
                    h = rect.bottom()

                shape = predictor(gray, rect)
                shape_np = face_utils.shape_to_np(shape).tolist()
                left_eye = midpoint(shape_np[36], shape_np[39])
                right_eye = midpoint(shape_np[42], shape_np[45])
                features = [left_eye, right_eye, shape_np[33], shape_np[48], shape_np[54]]   

                with open(pimg.split('.')[0] + ".txt", "a") as f:
                    for i in features:
                        print(str(i[0]) + ' ' + str(i[1]), file=f)   
                        
def midpoint(p1, p2):
    
    coords = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
	
    return [int(x) for x in coords]

if __name__ == '__main__':
    
    path = 'in/'
        
    createTxt('in/')