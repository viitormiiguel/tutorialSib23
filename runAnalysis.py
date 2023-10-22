
import cv2
import os 
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from feat.detector import Detector
from feat.data import Fex
from feat.utils.io import get_test_data_path
from feat.plotting import imshow

from runPlot import cropImage
import seaborn as sns


# Raiva 4, 5, 7, 10, 17, 22, 23, 24, 25, 26
# Medo 1, 2, 4, 5, 20, 25, 26
# Nojo 9, 10, 16, 17, 25, 26
# Felicidade 6, 12
# Tristeza 1, 4, 11, 15, 17
# Surpresa 1, 2, 5, 26

def openFace(image, tipo):
    
    path_img = path + '\\output\\'+tipo+'\\'
            
    exec1 = path + '\\output\\'+tipo+'\\' + image
    exec2 = path + '\\output\\openface\\' + image.split('.')[0]

    comando = ' -f ' + '"' +  exec1 + '"'

    os.system('"C:\\OpenFace\\FaceLandmarkImg.exe' + comando + ' -out_dir ' + exec2 + '"')
    
    time.sleep(1)

def dataOpen(image):
            
    nome = image.split('.')

    arq = path + '\\output\\openface\\' + image + '\\' + image + '.csv'
                    
    arquivo = pd.read_csv(arq)
            
    valores = arquivo.iloc[:, 676:693]
        
    return valores

def pyfeat(image, tipo):
    
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model='xgb',
        emotion_model="resmasknet",
        facepose_model="img2pose",
    )
    
    path_img = path + '\\output\\'+tipo+'\\'

    single_face_img_path = os.path.join(path_img, image)

    ## PY-FEAT
    single_face_prediction = detector.detect_image(single_face_img_path)

    emotions = single_face_prediction.emotions
    
    aus = single_face_prediction.aus
    
    return emotions, aus

def getInfoOpen(o1, o2):
    
    ret1 = []
    ret2 = []
    
    laus = list(o1.values)
    laus = list(laus[0])                
    lblAus = list(o1)
        
    getEmo = 5
    
    # print(ausEmo[getEmo])                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        name = ''
        if len(str(i)) == 1:
            name = ' AU0' + str(i) + '_r'
        else:
            name = ' AU' + str(i) + '_r'
        
        indice = lblAus.index(name)
        getVal = laus[indice]
        
        temp.append(getVal)
        
    ret2.append(temp)
    
    ## ============================================
    
    laus = list(o2.values)
    laus = list(laus[0])                
    lblAus = list(o2)
        
    getEmo = 5
    
    # print(ausEmo[getEmo])                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        name = ''
        if len(str(i)) == 1:
            name = ' AU0' + str(i) + '_r'
        else:
            name = ' AU' + str(i) + '_r'
        
        indice = lblAus.index(name)
        getVal = laus[indice]
        
        temp.append(getVal)
        
    ret2.append(temp)
    
    return ret1, ret2

def getInfos(p1, p2):
   
    ret1 = []
    ret2 = []

    emotions = p1[0]
    
    aus = p1[1]

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

    ## =========================================================

    emotions = p2[0]
    
    aus = p2[1]

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

def runPlots(retorno, retorno2):
    
    sns.set_theme()
    
    pimg = 'output/deep3d/'
    
    ## Deep3D Image
    img = cv2.imread(pimg + "photo0_mesh.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ## Crop Real Face    
    r = cropImage('photo0.png')
    
    plt.figure(figsize=(10,8))
        
    ## =========================================================            
    plt.subplot(3, 2, 1)
    plt.title('Real Face')
    plt.imshow(r)
    plt.axis('off')

    ## =========================================================
    plt.subplot(3, 2, 2)
    plt.title('Deep3D Face')
    plt.imshow(img)
    plt.axis('off')
    
    ## =========================================================    
    label1 = [str("AU") + str(x) for x in retorno[0][0]]
    valor1 = retorno[1][0]

    plt.subplot(3, 2, 3)
    plt.title('Intensities PyFeat')
    plt.ylim([0,1])
    plt.bar(label1, valor1)

    ## =========================================================    
    label2 = [str("AU") + str(x) for x in retorno[0][1]]
    valor2 = retorno[1][1]
    
    plt.subplot(3, 2, 4)
    plt.title('Intensities PyFeat')
    plt.ylim([0,1])
    plt.bar(label2, valor2)
    
    ## =========================================================    
    labelOpen1 = [str("AU") + str(x) for x in retorno2[0][0]]
    valorOpen1 = retorno2[1][0]

    plt.subplot(3, 2, 5)
    plt.title('Intensities OpenFace')
    plt.ylim([0,5])
    plt.bar(labelOpen1, valorOpen1)

    ## =========================================================    
    labelOpen2 = [str("AU") + str(x) for x in retorno2[0][1]]
    valorOpen2 = retorno2[1][1]
    
    plt.subplot(3, 2, 6)
    plt.title('Intensities OpenFace')
    plt.ylim([0,5])
    plt.bar(labelOpen2, valorOpen2)

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
    
    ## Run Openface
    # openFace('photo0.png', 'deep3d')
    # openFace('photo0_mesh.png', 'deep3d')
    
    ## Run Get data py-feat
    p1 = pyfeat('photo0.png', 'rec') 
    p2 = pyfeat('photo0_mesh.png', 'deep3d')    
        
    r = getInfos(p1, p2)
    
    ## Run Get data OpenFace
    o1 = dataOpen('photo0')
    o2 = dataOpen('photo0_mesh')
    
    s = getInfoOpen(o1, o2)
    
    runPlots(r, s)