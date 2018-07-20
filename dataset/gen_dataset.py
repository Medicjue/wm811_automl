# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd

src_folder = 'imgs'
poc_folder = 'poc_imgs'
categories = os.listdir(src_folder)
categories.remove('noLabel')
categories.remove('.DS_Store')

for category in categories:
    img_files = os.listdir(os.path.join(src_folder, category))
    train_img_files = img_files[0:241]
    test_img_files = img_files[241:482]
    dest_folder = os.path.join(poc_folder, 'TRAIN', category)
    os.makedirs(dest_folder, exist_ok=True)
    dest_folder = os.path.join(poc_folder, 'TEST', category)
    os.makedirs(dest_folder, exist_ok=True)
    if category == 'Near-full':
        train_img_files = img_files[0:74]
        test_img_files = img_files[74:149]
    if category == 'none':
        train_img_files = img_files[0:239]
        test_img_files = img_files[239:477]
    for train_img_file in train_img_files:
        src_file = os.path.join(src_folder, category, train_img_file)
        dest_file = os.path.join(poc_folder, 'TRAIN', category, train_img_file)
        shutil.copy(src_file, dest_file)
    for test_img_file in test_img_files:
        src_file = os.path.join(src_folder, category, test_img_file)
        dest_file = os.path.join(poc_folder, 'TEST', category, test_img_file)
        shutil.copy(src_file, dest_file)
        
### Generate Google AutoML all_data.csv
all_data = pd.DataFrame(columns=['TYPE', 'PATH', 'LABEL'])

types = ['TRAIN', 'TEST']

for dtype in types:
    categories = os.listdir(os.path.join(poc_folder, dtype))
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
    for category in categories:
        img_files = os.listdir(os.path.join(poc_folder, dtype, category))
        for img_file in img_files:
            record = dict()
            index = img_files.index(img_file)
            
            path = 'gs://tsmc-auto-ml-poc-vcm/wm811/poc_imgs/{}/{}/{}'.format(dtype, category, img_file)
            data_dtype = dtype
            if index > int(240*0.9) and dtype == 'TRAIN':
                data_dtype = 'VALIDATION'
            record['TYPE'] = data_dtype
            record['PATH'] = path
            record['LABEL'] = category
            all_data = all_data.append(record, ignore_index=True)
all_data = all_data.sort_values(by='TYPE').reset_index(drop=True)
all_data.to_csv('google_all_data.csv', index=False, header=False)

len(all_data[all_data['TYPE'] == 'TEST'])

