
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

from runCropFace import cropImage
import seaborn as sns
from sklearn import preprocessing as pre

# Raiva 4, 5, 7, 10, 17, 22, 23, 24, 25, 26
# Medo 1, 2, 4, 5, 20, 25, 26
# Nojo 9, 10, 16, 17, 25, 26
# Felicidade 6, 12
# Tristeza 1, 4, 11, 15, 17
# Surpresa 1, 2, 5, 26

def dataOpen(image, tipo):
            
    nome = image.split('.')

    arq = path + '\\out\\openface\\' + tipo + '\\' + image + '\\' + image + '.csv'
                    
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
    
    path_img = path + '\\out\\'+tipo+'\\'

    single_face_img_path = os.path.join(path_img, image)

    ## PY-FEAT
    single_face_prediction = detector.detect_image(single_face_img_path)

    emotions = single_face_prediction.emotions
    
    aus = single_face_prediction.aus
    
    # List emotions
    lemo = list(emotions.values)
    lemo = list(lemo[0])
    
    maximo = max(lemo)
    getEmo = lemo.index(maximo)
    
    return emotions, aus, getEmo

def getInfoOpen(o1, o2, o3, emo):
    
    ret1 = []
    ret2 = []
    
    laus = list(o1.values)
    laus = list(laus[0])                
    lblAus = list(o1)
        
    getEmo = 2
    
    # print(ausEmo[getEmo])                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        try: 
            
            name = ''
            if len(str(i)) == 1:
                name = ' AU0' + str(i) + '_r'
            else:
                name = ' AU' + str(i) + '_r'
            
            indice = lblAus.index(name)
            getVal = laus[indice]
            
            temp.append(getVal)
            
        except ValueError: 
            pass        
        
    ret2.append(temp)
    
    ## ============================================
    
    laus = list(o2.values)
    laus = list(laus[0])                
    lblAus = list(o2)
        
    getEmo = 2
    
    # print(ausEmo[getEmo])                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        try:
            
            name = ''
            if len(str(i)) == 1:
                name = ' AU0' + str(i) + '_r'
            else:
                name = ' AU' + str(i) + '_r'
            
            indice = lblAus.index(name)
            getVal = laus[indice]
            
            temp.append(getVal)
            
        except ValueError:
            pass
        
    ret2.append(temp)
    
    ## ============================================
    
    laus = list(o3.values)
    laus = list(laus[0])                
    lblAus = list(o3)
        
    getEmo = 2
    
    # print(ausEmo[getEmo])                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        try:
            
            name = ''
            if len(str(i)) == 1:
                name = ' AU0' + str(i) + '_r'
            else:
                name = ' AU' + str(i) + '_r'
            
            indice = lblAus.index(name)
            getVal = laus[indice]
            
            temp.append(getVal)
            
        except ValueError:
            pass
        
    ret2.append(temp)
    
    return ret1, ret2

def getInfos(p1, emo):
   
    ret1 = []
    ret2 = []

    emotions = p1[0]    
    aus = p1[1]
         
    # List emotions
    lemo = list(emotions.values)
    lemo = list(lemo[0])
                    
    # List aus
    laus = list(aus.values)
    laus = list(laus[0])                
    lblAus = list(aus)
        
    x = np.array(laus)
    x = x.reshape(-1, 1)

    #normalize all values to be between 0 and 1
    x_norm = pre.MinMaxScaler(feature_range=(0, 5)).fit_transform(x)

    maximo = max(lemo)
    getEmo = 2
                
    ret1.append(ausEmo[getEmo])
    
    temp = []                
    for i in ausEmo[getEmo]:
        
        name = ''
        if len(str(i)) == 1:
            name = 'AU0' + str(i)
        else:
            name = 'AU' + str(i)
        
        try :
            
            indice = lblAus.index(name)
            getVal = x_norm[indice][0]
            
            temp.append(getVal)
            
        except ValueError:
            pass;
        
    ret2.append(temp)
    
    return ret1, ret2

def runPlots(r1, r2, r3, retorno2):
        
    sns.set_theme()
    sns.set_style("dark")
    
    pimg = 'out/deep3d_pos/'
    
    ## Deep3D Image
    img = cv2.imread(pimg + f2, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ## Crop Real Face    
    r = cropImage(f1, 'real')
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    
    ## Crop Real Face    
    # retDeca = cropImage(f3, 'deca_pos')
    retDeca = cv2.imread('out/deca_pos/' + f3, cv2.COLOR_BGR2RGB)
    retDeca = cv2.cvtColor(retDeca, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(18,8))
        
    ## =========================================================            
    plt.subplot(2, 3, 1)
    plt.title('Real Face')
    plt.imshow(r)
    plt.axis('off')

    ## =========================================================
    plt.subplot(2, 3, 2)
    plt.title('Deep3D Face')
    plt.imshow(img)
    plt.axis('off')
    
    ## =========================================================
    plt.subplot(2, 3, 3)
    plt.title('DECA Face')
    plt.imshow(retDeca)
    plt.axis('off')
       
    ## =========================================================    
    # label1 = [str("AU") + str(x) for x in r1[0][0]]
    # valor1 = r1[1][0]

    # plt.subplot(3, 3, 4)
    # plt.title('Intensities PyFeat')
    # plt.ylim([0,5])
    # plt.bar(label1, valor1)

    # label2 = [str("AU") + str(x) for x in r2[0][0]]
    # valor2 = r2[1][0]
    
    # plt.subplot(3, 3, 5)
    # plt.title('Intensities PyFeat')
    # plt.ylim([0,5])
    # plt.bar(label2, valor2)
    
    # label3 = [str("AU") + str(x) for x in r3[0][0]]
    # valor3 = r3[1][0]
    
    # plt.subplot(3, 3, 6)
    # plt.title('Intensities PyFeat')
    # plt.ylim([0,5])
    # plt.bar(label3, valor3)
    
    ## =========================================================    
    labelOpen1 = [str("AU") + str(x) for x in retorno2[0][0]]
    valorOpen1 = retorno2[1][0]

    plt.subplot(2, 3, 4)
    plt.title('Intensities OpenFace')
    plt.ylim([0,5])
    plt.bar(labelOpen1, valorOpen1)

    ## =========================================================    
    labelOpen2 = [str("AU") + str(x) for x in retorno2[0][1]]
    valorOpen2 = retorno2[1][1]
    
    plt.subplot(2, 3, 5)
    plt.title('Intensities OpenFace')
    plt.ylim([0,5])
    plt.bar(labelOpen2, valorOpen2)
    
    ## =========================================================    
    labelOpen3 = [str("AU") + str(x) for x in retorno2[0][2]]
    valorOpen3 = retorno2[1][2]
    
    plt.subplot(2, 3, 6)
    plt.title('Intensities OpenFace')
    plt.ylim([0,5])
    plt.bar(labelOpen3, valorOpen3)
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.6)

    # plt.show()
    plt.savefig('output/results/results_' + f1)
    
if __name__ == '__main__': 
    
    path = 'E:\\PythonProjects\\tutorialSib23'
            
    labelEmo = ['anger','disgust', 'fear','happiness','sadness','surprise']
    
    ausEmo = [
        [4, 5, 7, 10, 17, 23, 25, 26],
        [9, 10, 17, 25, 26],
        [1, 2, 4, 5, 20, 25, 26],
        [6, 12],
        [1, 4, 15, 17],
        [1, 2, 5, 26],
    ]
    
    deep3dFiles = os.listdir('out/deep3d_pos')
    decaFiles   = os.listdir('out/deca_pos')
    realFiles   = os.listdir('out/real')
    
    for i, j in enumerate(realFiles):
        
        if 'fear1.jpg' in j:
    
            f1 = j
            f2 = deep3dFiles[i]
            f3 = decaFiles[i]
                    
            ## Run Get data py-feat
            p1 = pyfeat(f1, 'real') 
            p2 = pyfeat(f2, 'deep3d_pos')
            p3 = pyfeat(f3, 'deca_pos')
                        
            r1 = getInfos(p1, p1[2])
            r2 = getInfos(p2, p1[2])
            r3 = getInfos(p3, p1[2])
            
            ## Run Get data OpenFace
            o1 = dataOpen(f1.split('.')[0], 'real')
            o2 = dataOpen(f2.split('.')[0], 'deep3d')
            o3 = dataOpen(f3.split('.')[0], 'deca')
                    
            s = getInfoOpen(o1, o2, o3, p1[2])
            
            runPlots(r1, r2, r3, s)
            
            # break