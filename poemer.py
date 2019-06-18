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
import keras.backend as K

class KerasSession:
    def __enter__(self):
        #メモリを動的に拡張
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
    def __exit__(self, ex_value, ex_type, trace):
        #繰り返し実行するとメモリリークするので、セッションを一旦クリアする
        K.clear_session()

_filename = 'poem.csv'

class Maro:
    def __init__(self,filename=_filename,grams=3):
        self.filename = filename
        self.grams = grams

    def build(self, in_shape, out_size, layers=4, dropout=0.2, hidden=256):
        input_raw = Input(shape=in_shape)
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
                #activation='relu',
                #kernel_initializer='he_uniform',
                ))(x)
        label = Dense(
            units=out_size*7,
            activation='relu',
            kernel_initializer='he_uniform',
            )(x)
        label = Dense(units=out_size)(x)
        output = Activation('softmax')(label)
        model = Model(inputs=input_raw,outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

    def generate(self):
        poems = self._read()
        l2n_map, n2l_map = self._map(poems)
        n_poems = self._vectorize(l2n_map,poems)
        x = []
        y = []
        for poem in n_poems:
            for k in range(self.grams,len(poem)):
                x.append(poem[k-self.grams:k])
                y.append(poem[k])

        X = to_categorical(x)
        Y = to_categorical(y)
        return X,Y

    def training(self,model,X,Y):
        early_stopping = EarlyStopping(patience=50, verbose=1)
        history = model.fit(
            X, Y,
            epochs=120,
            batch_size=64,
            validation_split=0.2,
            shuffle=False,
            verbose=1,
            callbacks=[early_stopping])
        return history

    def describe(self,model,letters='#######'):
        poems = self._read()
        l2n_map, n2l_map = self._map(poems)

        poem = list(letters)
        while True:
            d = [ l2n_map[x] for x in poem[-self.grams:] ]
            X = to_categorical(d,num_classes=len(l2n_map))
            X = np.reshape(X,(1,*X.shape))
            Y = model.predict(X)
            y = n2l_map[self._prob(Y[0])]
            if y=='@':
                break
            else:
                poem.append(y)
        return ''.join(poem).replace('#','')

    def _prob(self,x,amp=0.8):
        Y = np.asarray(x).astype('float64')
        #Y[Y<1e-10]=1e-10
        Y = np.exp(np.log(Y)/amp)
        Y = Y/np.sum(Y)
        Y = np.random.multinomial(1,Y,1)
        return np.argmax(Y)

    def download(self,url='http://lapis.nichibun.ac.jp/waka/waka_i072.html'):
        pd.set_option("display.max_colwidth", 300)
        html = pd.read_html(url)
        poems = []
        for line in html[3:]:
            poem = str(line).split()[6]
            poem = poem.replace('−','').replace('.','')
            if not 'х' in poem:
                poems.append(poem)
        with open(self.filename,'w',encoding='UTF-8',newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(poems)

    def _read(self):
        poems = []
        with open(self.filename,'r',encoding='utf-8') as fp:
            reader = csv.reader(fp)
            for line in reader:
                poem = list('#'*self.grams)+line+list('@')
                poems.append(poem)
        return poems

    def _map(self,poems):
        letters = set()
        for poem in poems:
            letters |= set(poem)
        l2n_map = {letter:index for index,letter in enumerate(sorted(letters))}
        n2l_map = {index:letter for index,letter in enumerate(sorted(letters))}
        return l2n_map, n2l_map

    def _maxlength(self,poem):
        length = 0
        for poem in poems:
            length = max(length,len(poem))
        return length

    def _vectorize(self,l2n_map,poems):
        output = []
        for poem in poems:
            d = []
            for k,x in enumerate(poem):
                d.append(l2n_map[x])
            output.append(d)
        return output

    def save(self,model):
        model.save(self.filename+'.h5')
    def load(self):
        model=load_model(self.filename+'.h5')
        return model


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-t','--training',action='store_true')
    parser.add_argument('-c','--cont',action='store_true')
    parser.add_argument('-u','--update_csv',action='store_true')
    parser.add_argument('-i','--intro',nargs='?',type=str,const='',default='')
    parser.add_argument('-f','--filename',type=str,default='maro.csv')
    args = parser.parse_args()

    grams = 5
    m = Maro(filename=args.filename,grams=grams)
    if args.update_csv:
        m.download()

    elif args.training:
        X,Y = m.generate()
        with KerasSession() as ks:
            model = m.build(X.shape[1:],Y.shape[1])
            m.training(model,X,Y)
            m.save(model)

    elif args.cont:
        X,Y = m.generate()
        with KerasSession() as ks:
            model = m.load()
            m.training(model,X,Y)
            m.save(model)

    else:
        letters = args.intro
        if len(letters)<grams:
            letters = '#'*(grams-len(letters))+letters
        else:
            letters = letters[:grams]
        with KerasSession() as ks:
            model = m.load()
            for k in range(20):
                poem = m.describe(model,letters=letters)
                print(poem)
