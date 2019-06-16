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
        #poem = '#'+poem.replace('\n','').replace('.','') +'@'
        poem = '#'+poem.replace('−','').replace('.','') +'@'
        if not 'х' in poem:
            poems.append(poem)
    with open(filename,'w',encoding='UTF-8',newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(poems)


def read(filename=_filename):
    with open(filename,'r',encoding='utf-8') as fp:
        raw_data = list(csv.reader(fp))
    return raw_data

def l2n(data):
    letters = set()
    length = 0
    for line in data:
        letters |= set(line)
        length = max(length,len(line))
    dictionary = {letter:index for index,letter in enumerate(sorted(letters))}

    output = []
    for line in data:
        d = []
        for k,x in enumerate(line):
            d.append(dictionary[x])
        else:
            d.extend([dictionary['@']]*(length-k-1))
            output.append(d)
    return output, dictionary


def generate(line,grams=3):
    x = []
    y = []
    for k in range(grams,len(line)-1):
        x.append(line[k-grams:k])
        y.append(line[k+1])
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
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

def training(model,X,Y):
    early_stopping = EarlyStopping(patience=50, verbose=1)
    history = model.fit(
        X, Y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping])
    return history

def Maro_describe(letters,loop=1):
    raw_data = read()
    num_data, dictionary = l2n(raw_data)
    inv = {y:x for x,y in dictionary.items()}

    filename='maro.h5'
    model = load_model(filename)

    poem = list(letters)
    temp = 0.3
    while True:
        d=[]
        for x in poem[-len(letters):]:
            d.append(dictionary[x])
        X = to_categorical(d,num_classes=len(dictionary))
        X = np.reshape(X,(1,*X.shape))
        Y = model.predict(X)
        Z = np.asarray(Y[0]).astype('float64')
        Z[Z<1e-10]=1e-10
        Z = np.exp(np.log(Z)/temp)
        Z = np.random.multinomial(1,Z/np.sum(Z),1)
        y = inv[np.argmax(Z)]
        if y=='@':
            break
        else:
            poem.append(y)
    return poem

def Maro_learn(new=True):
    grams = 4
    filename='maro.h5'
    raw_data = read()
    num_data, dictionary = l2n(raw_data)
    X=[]
    Y=[]
    for line in num_data:
        x,y = generate(line, grams=grams)
        X.append(x)
        Y.append(y)

    X = to_categorical(X)
    Y = to_categorical(Y)
    X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2],X.shape[3]))
    Y = np.reshape(Y,(Y.shape[0]*Y.shape[1],Y.shape[2]))
    if new:
        model = build(in_size=(X.shape[1],X.shape[2]),out_size=Y.shape[1])
    else:
        model = load_model(filename)

    history = training(model,X,Y)
    model.save(filename)

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-t','--training',action='store_true')
    parser.add_argument('-c','--cont',action='store_true')
    parser.add_argument('-u','--update_csv',action='store_true')
    parser.add_argument('-i','--intro',nargs='?',type=str,const='あしひきの',default='やまさとは')
    args = parser.parse_args()

    if args.update_csv:
        download()
    elif args.training:
        Maro_learn(new=True)
    elif args.cont:
        Maro_learn(new=False)
    else:
        letters = '#'+args.intro[:3]
        poem = Maro_describe(letters)
        print(''.join(poem))
