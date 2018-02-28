# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from sklearn.naive_bayes import MultinomialNB
 
s = 1.
def new_score(p, s):
    if p[1] > 0.5:
        if p[1] < 0.5: return [0,0,0,1,0,0]
        else: return [0, 0, 0, 0, 1, 0]
    else:
        if p[0] > s:
            return [1, 0, 0, 0, 0, 0]
        else:
            if p[0] < 0.5: return [0, 0, 1, 0, 0, 0]
            else: return [0, 1, 0, 0, 0, 0]
 
class Classifier(BaseEstimator):
    def __init__(self):
#         self.clf2 = LogisticRegression(C=0.04)
        pass
    
    def model_full_connect(self,input_dim):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model
    
 
    def fit(self, X, y):
        dummy_y = np_utils.to_categorical((y>2).astype(int))
        self.clf1 = self.model_full_connect(X.shape[1])
        self.clf1.fit(X, dummy_y, epochs=15,batch_size=128)
 
    def predict(self, X):
        y_pred = self.clf1.predict_proba(X)
        return np.array([np.argmax(new_score(p,s)) for p in y_pred])
 
    def predict_proba(self, X):
        y_pred = self.clf1.predict_proba(X)
        return np.array([new_score(p, s) for p in y_pred])
    