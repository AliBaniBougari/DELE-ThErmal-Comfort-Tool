import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import Dense,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler
import pickle

#load data

def load_data(file_data,file_label):  
    with open(file_data, 'rb') as fp:
        data_set=pickle.load(fp)
    with open(file_label, 'rb') as fp:
        label=pickle.load(fp)
    return data_set,np.array(label).astype('float32')
#reshape 
def reshape(data):
    all_data = []
    for i in data:
        m = []
        for k in list(i):
            m.append(k)
        all_data.append([m])
    return np.array(all_data).astype('float32')

#test network

def test_net(X_test,y_test):
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

#plot history of net_work
    
def plot_history(history):
    a = history.history['accuracy']
    l = history.history['val_accuracy']
    plt.plot(a)
    plt.plot(l)
    plt.legend(['accuracy','val_accuracy'])
    plt.show()

#function for build model
    
def build_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(1,9),activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(128,activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(256,activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(256,activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(512,activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(512,activation='relu',return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    return model



data,label = load_data('all_data','all_label')

#Data standardization

scale = StandardScaler()

scale = scale.fit(data)

data_set = scale.transform(data)




#reshape data

r_data = reshape(data_set)

print(r_data.shape)

#split_data

X_train, X_test, y_train, y_test = train_test_split( r_data,label, test_size=0.1, shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, shuffle=True)

#build model

model = build_model()

#compile model

model.compile(optimizer = 'adam'
              ,loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


model.summary()

history = model.fit(X_train, y_train,verbose = 1 ,shuffle=True, epochs=250, batch_size=2000 , validation_data=(X_val,y_val))

test_net(X_test,y_test)

plot_history(history)
