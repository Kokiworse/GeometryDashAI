from grabscreen import grab_screen
from model import geometrynet
from keras.models import load_model
import cv2
import numpy as np


WIDTH = 120
HEIGHT = 160
model = load_model('model_test.h5')

def jump():
    pressKey(W)
    releaseKey(W)

def main():
    paused = False

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    while(True):

        if not paused:
            screen = grab_screen(region=(0,40,800,640))

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))  #input for the nn as small as possible so its easy to process
            cv2.imshow('Geometry dash', screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            prediction = model.predict([screen.reshape(-1,WIDTH,HEIGHT,1)])[0]
            prediction = np.array(prediction)

            action = np.argmax(prediction)

            if(action == 1):
                jump()
        if 'P' in keys: #press p to pause the game
            if paused:
                paused = False
                print('unpaused')
                time.sleep(1)
            else:
                print('paused')
                paused = True
                time.sleep(1)
