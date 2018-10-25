import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2


train_data = np.load('training_data.npy')
# print(len(train_data))
shuffle(train_data)

jump = []
nojump = []

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == 1:
        jump.append([img,choice])

    elif choice == 0:
        nojump.append([img,choice])


nojump = nojump[:len(jump)]

dataset = jump + nojump
shuffle(dataset)

np.save('data_train', dataset)
