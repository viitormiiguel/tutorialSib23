
import dlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
from PIL import Image

from simple_3dviz import Mesh
from simple_3dviz.window import show
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames

import shutil

import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')


def convertImg():
    
    lista = os.listdir('test_out')

    for l in lista:
        
        if '.obj' in l:
            print("test_out/" + l)
            m = Mesh.from_file("test_out/" + l, color=(0.8, 0.8, 0.8, 1.0))
            m.to_unit_cube()
            
            nome = l.split('.')
            # print(nome)
            render(
                [m],
                n_frames=200,
                size=(256,256),
                camera_position=(0.0, 0.15, 1.5),
                up_vector=(0, 1, 0),
                behaviours=[
                    SaveFrames('test_out/' + nome[0] + '.png')
                ]
            )
            
def copyImg():
    
    lista = os.listdir('out/deca')

    for l in lista:
        
        if '.jpg' not in l:
            
            images = os.listdir('out/deca/' + l)
            
            for i in images:
                
                if i == str(l + '_rendered_images.jpg'):
                    
                    original = 'out/deca/' + l + '/' + i
                    target = 'out/deca_pos/' + i
                    
                    shutil.copyfile(original, target)    
            
if __name__ == '__main__':
    
    path = 'in/'
        
    convertImg()
    
    # copyImg()