
import cv2
import os 
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import seaborn as sns
from sklearn import preprocessing as pre

def chartPlot(ret, nome, legenda):

    sns.set_theme()
    # sns.set_style("dark")

    inflationAndGrowth = {
        "Deepd3D": [ -i for i in ret[0] ],
        "Deca": [ -i for i in ret[1] ]
    };

    index  = [str("AU") + str(x) for x in legenda]

    dataFrame = pd.DataFrame(data = inflationAndGrowth);

    dataFrame.index = index;

    dataFrame.plot.barh(rot=15, title="Intensity differences between Real vs Deep3D and Deca");

    # plt.show(block=True);
    
    plt.savefig('output/results/dif_' + nome + '.jpg')
    
def getData(name):
    
    pathOpen = 'out/openface/'
    
    pathOpenReal = pathOpen + 'real/' + name + '/' + name + '.csv'
    pathOpenDeep = pathOpen + 'deep3d/' + name + '_mesh/' + name + '_mesh.csv'
    pathOpenDeca = pathOpen + 'deca/' + name + '_rendered_images/' + name + '_rendered_images.csv'
    
    arq1 = pd.read_csv(pathOpenReal)
    arq2 = pd.read_csv(pathOpenDeep)
    arq3 = pd.read_csv(pathOpenDeca)
            
    val1 = arq1.iloc[:, 676:693].values
    val2 = arq2.iloc[:, 676:693].values
    val3 = arq3.iloc[:, 676:693].values
    
    return val1, val2, val3    

def calculate(ret, indice):
    
    lista1 = list(ret[0][0])    
    lista2 = list(ret[1][0])
    lista3 = list(ret[2][0])
    
    result1 = []    
    for i in range(len(lista1)):
        result1.append(lista1[i] - lista2[i])
        
    result2 = []    
    for i in range(len(lista1)):
        result2.append(lista1[i] - lista3[i])
        
    tmp1 = []
    for ind in indice:
        tmp1.append(result1[ind])
        
    tmp2 = []
    for ind in indice:
        tmp2.append(result2[ind])
        
    # print('result deep3', result1)
    # print('result deca', result2)
    
    return tmp1, tmp2
    
if __name__ == '__main__': 
    
    nome = 'sad2'
    r = getData(nome)
    
    labelEmo = ['anger','disgust', 'fear','happiness','sadness','surprise']
    
    ausEmo = [
        [4, 5, 7, 10, 17, 23, 25, 26],
        [9, 10, 17, 25, 26],
        [1, 2, 4, 5, 20, 25, 26],
        [6, 12],
        [1, 4, 15, 17],
        [1, 2, 5, 26],
    ]
    
    aus = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
    
    indice = []
    for emo in ausEmo[5]:
        ind = aus.index(emo)
        indice.append(ind)
        
    ret = calculate(r, indice)
    
    chartPlot(ret, nome, ausEmo[5])