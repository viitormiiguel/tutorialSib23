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

def openFaceReal(image, way):
                
    exec1 = path + '\\in\\' + image
    exec2 = path + '\\out\\openface\\' + way + '\\' + image.split('.')[0]

    comando = ' -f ' + '"' +  exec1 + '"'

    os.system('"C:\\OpenFace\\FaceLandmarkImg.exe' + comando + ' -out_dir ' + exec2 + '"')
    
    time.sleep(1)

def openFace(image, tipo, way):
                
    exec1 = path + '\\out\\' + tipo + '\\' + image
    exec2 = path + '\\out\\openface\\' + way + '\\' + image.split('.')[0]

    comando = ' -f ' + '"' +  exec1 + '"'

    os.system('"C:\\OpenFace\\FaceLandmarkImg.exe' + comando + ' -out_dir ' + exec2 + '"')
    
    time.sleep(1)
    
if __name__ == '__main__': 
    
    path = 'E:\\PythonProjects\\tutorialSib23'
    
    way = ['out/deep3d_pos', 'out/deca_pos']
    out = ['deep3d', 'deca']
    
    ## CG Faces
    for i, j in enumerate(way):        
        for l in os.listdir(j):            
            if '.txt' not in l:                
                openFace(l, j.split('/')[1], out[i])
                
    ## Real images
    for l in os.listdir('in'):                    
        if '.txt' not in l:            
            openFaceReal(l, 'real')
            