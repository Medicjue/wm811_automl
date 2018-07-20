#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 02:17:20 2018

Convert all data waferMap into pngs
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('LSWMD.pkl')


all_data = pd.DataFrame(columns=['TYPE', 'PATH', 'LABEL'])


for index, row in data.iterrows():
    trainTestLabel = row['trianTestLabel']
    if type(trainTestLabel) == np.ndarray:
        if len(trainTestLabel) == 0:
            trainTestLabel = 'noLabel'
        else:
            trainTestLabel = trainTestLabel[0]
    if type(trainTestLabel) == np.ndarray:
        if len(trainTestLabel) == 0:
            trainTestLabel = 'noLabel'
        else:
            trainTestLabel = trainTestLabel[0]
    
    label = row['failureType']
    if type(label) == np.ndarray:
        if len(label) == 0:
            label = 'noLabel'
        label = label[0]
    if type(label) == np.ndarray:
        if len(label) == 0:
            label = 'noLabel'
        label = label[0]
    if 'n' == label:
        label == 'noLabel' 
    img = row['waferMap']
    os.makedirs('imgs/{}/'.format(trainTestLabel), exist_ok=True)
    os.makedirs('imgs/{}/{}/'.format(trainTestLabel, label), exist_ok=True)
    img_file_name = 'img_{}.png'.format(index)
    plt.imsave('imgs/{}/{}/{}'.format(trainTestLabel, label, img_file_name), img)
    
    if trainTestLabel == 'noLabel':
        continue
    elif trainTestLabel == 'Training':
        trainTestLabel = 'TRAIN'
    else:
        trainTestLabel = 'TEST'
    
    record = dict()
    path = 'gs://tsmc-auto-ml-poc-vcm/wm811/imgs/{}/{}/{}'.format(trainTestLabel, label, img_file_name)
    record['TYPE'] = trainTestLabel
    record['PATH'] = path
    record['LABEL'] = label
    all_data = all_data.append(record, ignore_index=True)
all_data = all_data.sort_values(by='TYPE').reset_index(drop=True)
all_data.to_csv('google_all_data.csv', index=False, header=False)

''''''
import pandas as pd
all_data = pd.read_csv('dataset/google_all_data.csv', names=['TYPE', 'PATH', 'LABEL'])

valid_indices = []

train_data = all_data[all_data['TYPE']=='TRAIN']
labels = set(train_data['LABEL'].tolist())
for l in labels:
    label_train_data = train_data[train_data['LABEL']==l]
    cut_pnt = int(len(label_train_data)*0.9)
    verified_train_data = label_train_data.iloc[cut_pnt:,:]
    valid_indices += verified_train_data.index.tolist()
    
for index, row in all_data.iterrows():
    if index in valid_indices:
        all_data.at[index, 'TYPE'] = 'VALIDATION'
        
all_data.to_csv('dataset/google_all_data_all.csv', index=False, header=False)

for index, row in all_data.iterrows():
    if row['TYPE'] == 'TRAIN' or row['TYPE'] == 'VALIDATION':
        new_path = row['PATH'].replace('TRAIN', 'Training')
        all_data.at[index, 'PATH'] = new_path
    else:
        new_path = row['PATH'].replace('TEST', 'Test')
        all_data.at[index, 'PATH'] = new_path
                   
                   
                   