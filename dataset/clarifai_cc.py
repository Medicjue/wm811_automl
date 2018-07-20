# -*- coding: utf-8 -*-

import os
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


app = ClarifaiApp(api_key='af9120c8d83549e48055ceadbfa9e531')

poc_folder = 'poc_imgs'

types = ['TRAIN']


for dtype in types:
    categories = os.listdir(os.path.join(poc_folder, dtype))
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')
    for category in categories:
        img_files = os.listdir(os.path.join(poc_folder, dtype, category))
        for img_file in img_files:
            f = os.path.join(poc_folder, dtype, category, img_file)
            not_categories = categories[:]
            not_categories.remove(category)
            image = ClImage(file_obj=open(f, 'rb'), concepts=[category], not_concepts=not_categories)
            resp = app.inputs.bulk_create_images([image])

model = app.models.create('wm811_benchmark', concepts=categories)

model = app.models.get('wm811_benchmark')
model.train()


model = app.models.get('wm811_benchmark')
ans = []
predicts = []
dtype = 'TEST'
categories = os.listdir(os.path.join(poc_folder, dtype))
if '.DS_Store' in categories:
    categories.remove('.DS_Store')
for category in categories:
    img_files = os.listdir(os.path.join(poc_folder, dtype, category))
    for img_file in img_files:
        f = os.path.join(poc_folder, dtype, category, img_file)
        image = ClImage(file_obj=open(f, 'rb'))
        ans.append(category)
        resp = model.predict([image])
        concepts = resp['outputs'][0]['data']['concepts']
        cc = []
        for concept in concepts:
            cc.append((concept['name'], concept['value']))
        cc = sorted(cc, key=lambda tup: tup[1], reverse=True)
        predicts.append(cc[0][0])
        
cf = confusion_matrix(ans, predicts, labels=categories)
score = accuracy_score(ans, predicts)


cf_df = pd.DataFrame(cf)
cf_df.columns = categories
cf_df.index = categories
cf_df.to_csv('clarifai_confusion_matrix.csv')
#
#img_file = img_files[2]
#f = os.path.join(poc_folder, dtype, category, img_file)
#image = ClImage(file_obj=open(f, 'rb'))
#model.predict([image])
#
#            app.inputs.create_image_from_url(url=f, concepts=[category])
#            path = 'gs://tsmc-auto-ml-poc-vcm/wm811/poc_imgs/{}/{}/{}'.format(dtype, category, img_file)
#            record['TYPE'] = dtype
#            record['PATH'] = path
#            record['LABEL'] = category
#            all_data = all_data.append(record, ignore_index=True)
#
#app.inputs.create_image_from_url(url='https://samples.clarifai.com/puppy.jpeg', concepts=['my puppy'])