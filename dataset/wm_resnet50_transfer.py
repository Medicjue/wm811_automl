# -*- coding: utf-8 -*-
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam  
from keras.layers import Flatten, Dense
from keras.models import Model

import os
import numpy as np

import random
import sys
import datetime as dt
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score

def create_resnet50_model():
    base_model = ResNet50(include_top=False, input_shape=(224,224,3))
    
    for l in base_model.layers:
        l.trainable = False
    
    z = base_model.output
    z = Flatten()(z)
    z = Dense(4096, activation='relu')(z)
    z = Dense(4096, activation='relu')(z)
    predictions = Dense(9, activation='softmax')(z)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])  
    model.summary()
    return base_model, model

def predict_base_resnet50(model, X):
    predicts = []
    for x in X:
        predicts.append(model.predict(x))
    return predicts


def load_data_for_resnet50(folder):
    X = []
    Y = []
    labels = os.listdir(folder)
    if '.DS_Store' in labels:
        labels.remove('.DS_Store')
    for label in labels:
        imgs = os.listdir(os.path.join(folder, label))
        for img in imgs:
            y = np.zeros(len(labels), dtype=int)
            y[labels.index(label)] = 1
            Y.append(y)
            img_file = os.path.join(folder, label, img)
            x = image.load_img(img_file, target_size=(224, 224))
            x = image.img_to_array(x)
            x = preprocess_input(x)
            X.append(x)
    return X, Y, labels

def randomize_train_data(X, Y):
    indices = [i for i in range(len(X))]
    random.shuffle(indices)

    tmp = []
    for index in indices:
        tmp.append(X[index])
    X = tmp
    tmp = []
    for index in indices:
        tmp.append(Y[index])
    Y = tmp
    
    return X, Y


    

if __name__ == '__main__':
    f = open('wm_resnet50_log.txt', 'w')
    sys.stdout = f
    sys.stderr = f
    
    train_folder = 'imgs/Training/'
    train_X, train_Y, train_labels = load_data_for_resnet50(train_folder)
    train_X, train_Y = randomize_train_data(train_X, train_Y)
    
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)
    print(train_X.shape)
    print(train_Y.shape)
    base_model, model = create_resnet50_model()
    fit_start = dt.datetime.now()
    model.fit(train_X, train_Y, epochs=10, validation_split=0.1)
    fit_end = dt.datetime.now()
    print('Transfer Training Time: {}'.format(fit_end - fit_start))
    model.save('wm_resnet50_transfer.model')
    
    
    test_folder = 'imgs/Test/'
    test_X, test_Y, test_labels = load_data_for_resnet50(test_folder)
    
    print(test_X.shape)
    print(test_Y.shape)
    
    inf_start = dt.datetime.now()
    predict_Y = model.predict(test_X)
    predict_Y_classes = np.argmax(predict_Y, axis=1)
    inf_end = dt.datetime.now()
    print('Inference Time: {}'.format(inf_end - inf_start))
    test_Y_classes = np.argmax(test_Y, axis=1)    
    cf = confusion_matrix(test_Y_classes, predict_Y_classes)
    score = accuracy_score(test_Y_classes, predict_Y_classes)
    print('Confusion Matrix:')
    print(cf)
    print('Accuracy Score:')
    print(score)
    
    cf_df = pd.DataFrame(cf)
    cf_df.columns = test_labels
    cf_df.index = test_labels
    cf_df.to_csv('wm_resnet50_result.csv')
