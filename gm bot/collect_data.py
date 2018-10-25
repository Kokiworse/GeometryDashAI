import numpy as np
import cv2
import time
import os
from grabscreen import grab_screen
from getkeys import key_check


def get_output(keys):
    '''
    convert the key pressed to 1 (jump) or 0 (no jump)
    '''
    output = 0

    if 'W' in keys:
        output = 1
    else:
        output = 0

    return output

file_name = 'training_data.npy' #file name

#checking if the file already exists

if os.path.isfile(file_name):
    print('The file already exists')
    training_data = list(np.load(file_name))
else:
    print('The file does not exist')
    training_data = []

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
            cv2.imshow('ss', screen)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            keys = key_check()
            output = get_output(keys)
            training_data.append([screen,output])

            if(len(training_data) % 3000 == 0):
                # print(len(training_data))
                np.save(file_name, training_data)

        keys = key_check()
        if 'P' in keys: #press p to pause the data collection
            if paused:
                paused = False
                print('unpaused')
                time.sleep(1)
            else:
                print('paused')
                paused = True
                time.sleep(1)

main()
