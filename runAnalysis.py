
import cv2
import os 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
from feat.plotting import imshow

from runPlot import cropImage

# Raiva 4, 5, 7, 10, 17, 22, 23, 24, 25, 26
# Medo 1, 2, 4, 5, 20, 25, 26
# Nojo 9, 10, 16, 17, 25, 26
# Felicidade 6, 12
# Tristeza 1, 4, 11, 15, 17
# Surpresa 1, 2, 5, 26

def getInfos():

    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model='xgb',
        emotion_model="resmasknet",
        facepose_model="img2pose",
    )

    lista = os.listdir('output/rec/')
    for l in lista:
        
        if 'photo' in l:
            
            if '.jpg' in l or '.png' in l:
                                        
                path_img = path + '\\output\\rec\\'

                single_face_img_path = os.path.join(path_img, l)

                single_face_prediction = detector.detect_image(single_face_img_path)

                emotions = single_face_prediction.emotions
                
                aus = single_face_prediction.aus

                print(emotions)
                print(aus)
                
    lista = os.listdir('output/deep3d/')
    for l in lista:
        
        if 'photo' in l:
            
            if '.jpg' in l or '.png' in l:
                                        
                path_img = path + '\\output\\deep3d\\'

                single_face_img_path = os.path.join(path_img, l)

                single_face_prediction = detector.detect_image(single_face_img_path)

                emotions = single_face_prediction.emotions
                
                aus = single_face_prediction.aus

                print(emotions)
                print(aus)

def runPlots():
    
    pimg = 'output/deep3d/'
    
    img = cv2.imread(pimg + "photo0_mesh.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    r = cropImage('photo0.png')
  
    # img2 = cv2.imread(r)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.axis('off')

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(r)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img)

    plt.subplot(2, 2, 4)
    plt.imshow(img)

    plt.show()
    
if __name__ == '__main__': 
    
    path = 'E:\\PythonProjects\\tutorialSib23'
        
    labelEmo = ['Raiva','Medo','Nojo','Felicidade','Tristeza','Surpresa']
    
    ausEmo = [
        [4, 5, 7, 10, 17, 22, 23, 24, 25, 26],
        [1, 2, 4, 5, 20, 25, 26],
        [9, 10, 16, 17, 25, 26],
        [6, 12],
        [1, 4, 11, 15, 17],
        [1, 2, 5, 26],
    ]    
    
    # getInfos()
    
    runPlots()