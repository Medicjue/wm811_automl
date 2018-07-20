# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('LSWMD.pkl')

for index, row in data.iterrows():
    label = row['failureType']
    if type(label) == np.ndarray:
        if len(label) == 0:
            label = 'noLabel'
        label = label[0]
    if type(label) == np.ndarray:
        if len(label) == 0:
            label = 'noLabel'
        label = label[0]
    
    img = row['waferMap']
    os.makedirs('imgs/{}/'.format(label), exist_ok=True)
    plt.imsave('imgs/{}/img_{}.png'.format(label, index), img)
    
all_data = pd.DataFrame(columns=['TYPE', 'PATH', 'LABEL'])

labels = os.listdir('imgs/')
labels.remove('.DS_Store')
labels.remove('Near-full')

for label in labels:
    img_files = os.listdir(os.path.join('imgs', label))
    if len(img_files) > 188:
        img_files = img_files[0:188]
    index = 0
    for img in img_files:
        record = dict()
        path = 'gs://tsmc-auto-ml-poc-vcm/wm811/imgs/{}/{}'.format(label, img)
        if index < 113:
            dtype = 'TRAIN'
        elif index >= 113 and index < 125:
            dtype = 'VALIDATION'
        elif index >= 125:
            dtype = 'TEST'
        
        record['TYPE'] = dtype
        record['PATH'] = path
        record['LABEL'] = label
        all_data = all_data.append(record, ignore_index=True)
        index += 1
        
all_data = all_data.sort_values(by='TYPE').reset_index(drop=True)

all_data.to_csv('all_data.csv', index=False, header=False)
set(all_data['LABEL'].tolist())
