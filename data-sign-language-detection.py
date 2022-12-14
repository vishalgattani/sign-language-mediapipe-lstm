import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform
import mediapipe as mp
import cv2
import numpy as np
import time
import os

#import from .py files
from mp_utils import mediapipe_detection,draw_landmarks, mp_drawing, mp_holistic, captureLandmarks
from data_collection import dataCollection
# from lstm_model import model,loadModel

DATA_PATH = os.path.join('mp_data')

def printInfo():
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

def checkDataCollected(actions,num_sequences):
    if(os.path.exists(os.path.join(DATA_PATH))):
        flag = True
    else:
        flag = False
        return flag
    flag = False
    dirsExist = []
    for action in actions:
            for num_sequence in range(num_sequences):
                dirsExist.append((os.path.exists(os.path.join(DATA_PATH,action,str(num_sequence)))," : ",DATA_PATH,action,str(num_sequence)))
    sequencesExist = []
    if(all(dirsExist)):
        for action in actions:
            if(len(os.listdir(os.path.join(DATA_PATH,action,str(num_sequence))))==num_sequences):
                sequencesExist.append(True)
                print(f'{num_sequences} sequences present in {os.path.join(DATA_PATH,action)}')

    if(all(sequencesExist)):
        flag = True
    return flag

if __name__ == "__main__":
    # environment info
    printInfo()

    # check if mediapipe is functioning correctly
    test_captureLandmarks = input("Check if mediapipe is working? [y/n]: ")
    if test_captureLandmarks == 'y':
        captureLandmarks()

    # # define actions, number of sequences and number of frames for each sequence to be collected
    # actions = ["hello","thank you","call"]
    # num_sequences = 75
    # num_frames = 30

    # # check if data needs to be collected for actions
    # checkData = checkDataCollected(actions,num_sequences)
    # if not checkData:
    #     dataCollection(actions,num_sequences,num_frames)
    # else:
    #     print("Data available.")