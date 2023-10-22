
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
    
    ret1 = []
    ret2 = []

    lista = os.listdir('output/rec/')
    for l in lista:
        
        if 'photo0' in l:
            
            if '.jpg' in l or '.png' in l:
                                        
                path_img = path + '\\output\\rec\\'

                single_face_img_path = os.path.join(path_img, l)

                single_face_prediction = detector.detect_image(single_face_img_path)

                emotions = single_face_prediction.emotions
                
                aus = single_face_prediction.aus

                # print(emotions, type(emotions))
                # print(aus)                
                
                # List emotions
                lemo = list(emotions.values)
                lemo = list(lemo[0])
                                
                # List aus
                laus = list(aus.values)
                laus = list(laus[0])                
                lblAus = list(aus)

                maximo = max(lemo)
                getEmo = lemo.index(maximo)
                
                # print(ausEmo[getEmo])                
                ret1.append(ausEmo[getEmo])
                
                temp = []                
                for i in ausEmo[getEmo]:
                    
                    name = ''
                    if len(str(i)) == 1:
                        name = 'AU0' + str(i)
                    else:
                        name = 'AU' + str(i)
                    
                    indice = lblAus.index(name)
                    getVal = laus[indice]
                    
                    temp.append(getVal)
                    
                # print(temp)
                ret2.append(temp)

                
    lista = os.listdir('output/deep3d/')
    for l in lista:
        
        if 'photo0' in l:
            
            if '.jpg' in l or '.png' in l:
                                        
                path_img = path + '\\output\\deep3d\\'

                single_face_img_path = os.path.join(path_img, l)

                single_face_prediction = detector.detect_image(single_face_img_path)

                emotions = single_face_prediction.emotions
                
                aus = single_face_prediction.aus

                # print(emotions)
                # print(aus)
                
                # List emotions
                lemo = list(emotions.values)
                lemo = list(lemo[0])
                                
                # List aus
                laus = list(aus.values)
                laus = list(laus[0])                
                lblAus = list(aus)

                maximo = max(lemo)
                getEmo = lemo.index(maximo)
                
                # print(ausEmo[getEmo])                
                ret1.append(ausEmo[getEmo])
                
                temp = []                
                for i in ausEmo[getEmo]:
                    
                    name = ''
                    if len(str(i)) == 1:
                        name = 'AU0' + str(i)
                    else:
                        name = 'AU' + str(i)
                    
                    indice = lblAus.index(name)
                    getVal = laus[indice]
                    
                    temp.append(getVal)
                    
                # print(temp)
                ret2.append(temp)
    
    return ret1, ret2

def runPlots(retorno):
    
    pimg = 'output/deep3d/'
    
    ## Deep3D Image
    img = cv2.imread(pimg + "photo0_mesh.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ## Crop Real Face    
    r = cropImage('photo0.png')
        
    ## =========================================================            
    plt.subplot(2, 2, 1)
    plt.title('Real Face')
    plt.imshow(r)
    plt.axis('off')

    ## =========================================================
    plt.subplot(2, 2, 2)
    plt.title('Deep3D Face')
    plt.imshow(img)
    plt.axis('off')
    
    ## =========================================================    
    label1 = [str(x) for x in retorno[0][0]]
    valor1 = retorno[1][0]

    plt.subplot(2, 2, 3)
    plt.ylim([0,1])
    plt.bar(label1, valor1)

    ## =========================================================    
    label2 = [str(x) for x in retorno[0][1]]
    valor2 = retorno[1][1]
    
    plt.subplot(2, 2, 4)
    plt.ylim([0,1])
    plt.bar(label2, valor2)

    plt.show()
    
if __name__ == '__main__': 
    
    path = 'E:\\PythonProjects\\tutorialSib23'
        
    labelEmo = ['anger','disgust', 'fear','happiness','sadness','surprise']
    
    ausEmo = [
        [4, 5, 7, 10, 17, 22, 23, 24, 25, 26],
        [9, 10, 16, 17, 25, 26],
        [1, 2, 4, 5, 20, 25, 26],
        [6, 12],
        [1, 4, 11, 15, 17],
        [1, 2, 5, 26],
    ]    
    
    r = getInfos()
    
    runPlots(r)