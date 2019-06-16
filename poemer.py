#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import csv

from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional, Input, Multiply, Concatenate, SpatialDropout1D
from keras.layers.recurrent import LSTM, RNN, SimpleRNN, GRU
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model

_filename = 'poem.csv'
def download( url='http://lapis.nichibun.ac.jp/waka/waka_i072.html', filename=_filename):
    html = pd.read_html(url)
    poems = []
    for line in html[3:]:
        poem = str(line).split()[6]
        poem = '#'+poem.replace('−','').replace('.','') +'@'
        if not 'х' in poem:
            poems.append(poem)
    with open(filename,'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(poems)


def read(filename=_filename):
    with open(filename,'r') as fp:
        raw_data = list(csv.reader(fp))
    return raw_data

def l2n(data):
    letters = set()
    length = 0
    for line in data:
        letters |= set(line)
        length = max(length,len(line))
    letters = {letter:index for index,letter in enumerate(sorted(letters))}

    output = []
    for line in data:
        d = []
        for k,x in enumerate(line):
            d.append(letters[x])
        else:
            d.extend([letters['@']]*(length-k-1))
            output.append(d)
    return output


def generate(line,grams=3):
    x = []
    y = []
    for k in range(len(line)-grams-1):
        x.append(line[k:k+grams])
        y.append(line[k+grams])
    return(x,y)

def build(in_size,out_size,layers=4,dropout=0.2,hidden=256):
    input_raw = Input(shape=in_size)
    x = input_raw
    for k in range(layers):
        if k != layers-1:
            s = True
        else:
            s = False
        x = Dropout(dropout)(x)
        x = Bidirectional(LSTM(
            units=hidden,
            return_sequences=s,
            ))(x)
    label = Dense(units=out_size)(x)
    output = Activation('softmax')(label)
    model = Model(inputs=input_raw,outputs=output)
    model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
    return model

def training(model,X,Y):
    early_stopping = EarlyStopping(patience=50, verbose=1)
    history = model.fit(
        X, Y,
        epochs=1000,
        batch_size=64,
        validation_split=0.2,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping])
    return history

if __name__ == '__main__':
    #download()
    gram = 4
    raw_data = read()
    num_data = l2n(raw_data)
    X=[]
    Y=[]
    for line in num_data:
        x,y = generate(line, grams=gram)
        X.append(x)
        Y.append(y)

    X = to_categorical(X)
    Y = to_categorical(Y)
    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2],X.shape[3]))
    Y = np.reshape(Y,(Y.shape[0]*Y.shape[1],Y.shape[2]))
    model = build(in_size=(X.shape[1],X.shape[2]),out_size=Y.shape[1])
    history = training(model,X,Y)
    model.save('maro.h5')
