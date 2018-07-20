# -*- coding: utf-8 -*-
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam  
from keras.layers import Flatten, Dense
from keras.models import Model

import os
import numpy as np

import random

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
#            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X.append(x)
    return X, Y

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
    base_model, model = create_resnet50_model()
    train_folder = 'dataset/imgs/Training/'
    train_X, train_Y = load_data_for_resnet50(train_folder)
    train_X, train_Y = randomize_train_data(train_X, train_Y)
    
    import pickle
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    
    f = open('large_wm811k_train_X.list', 'wb')
    bytes_out = pickle.dumps(train_X)
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()
    
    f = open('large_wm811k_train_Y.list', 'wb')
    bytes_out = pickle.dumps(train_Y)
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()
    
    test_folder = 'dataset/imgs/Test/'
    test_X, test_Y = load_data_for_resnet50(test_folder)
    
    import pickle
    f = open('large_wm811k_test_X.list', 'wb')
    bytes_out = pickle.dumps(test_X)
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()
    
    f = open('large_wm811k_test_Y.list', 'wb')
    bytes_out = pickle.dumps(test_Y)
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()
    
#    train_X = np.asarray(train_X)
#    train_Y = np.asarray(train_Y)
#    print(train_X.shape)
#    print(train_Y.shape)
#    model.fit(train_X, train_Y, epochs=10, validation_split=0.1)
#    model.save('resnet_wm811.model')
#    
#    
#    print(test_X.shape)
#    print(test_Y.shape)
#    
#    predict_Y = model.predict(test_X)
#    predict_Y_classes = np.argmax(predict_Y, axis=1)
#    test_Y_classes = np.argmax(test_Y, axis=1)    
#    cf = confusion_matrix(test_Y_classes, predict_Y_classes)
#    score = accuracy_score(test_Y_classes, predict_Y_classes)
#    
#    labels = os.listdir(train_folder)
#    labels.remove('.DS_Store')
#    import pandas as pd
#    cf_df = pd.DataFrame(cf)
#    cf_df.columns = labels
#    cf_df.index = labels
#    cf_df.to_csv('confusion_matrix.csv')
