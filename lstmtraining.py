import sys,os
import numpy as np
if(not sys.version_info[0]<3):
    from importlib import reload 

n2cdic=np.load('n2c.npy')
c2ndic=np.load('c2n.npy')
c2ndic=c2ndic[()]
n2cdic=n2cdic[()]

cha=np.load('poems_input.npy')
dic=np.unique(cha)
ohe=np.eye(len(dic))
cha=cha[:np.floor(len(cha)/2).astype(int)]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam

dataX_L=[ohe[c2ndic[c]] for c in cha]

dot=ohe[c2ndic['.']]
timesteps=30
ts=timesteps-1
dataX=np.zeros((len(dataX_L),timesteps,len(dataX_L[0])))
for i in range(ts):
    dataX[i]=np.concatenate(([dot]*(ts-i),dataX_L[0:i+1]))
dataX[timesteps:]=np.array([dataX_L[i:i+timesteps] for i in range(len(dataX_L)-timesteps)])

y=dataX[1:,-1]
shpY=np.shape(y)

shp=np.shape(dataX)
#X = np.reshape(dataX[:-1], (shp[0]-1, timesteps, shp[1]))
X=dataX[:-1]
y=np.reshape(y,(shpY[0],shpY[1]))
#-------------------------------------------------------
#firstLSTMoutput=X.shape[2]
firstLSTMoutput=256

model = Sequential()
model.add(LSTM(firstLSTMoutput, input_shape=(X.shape[1], X.shape[2]),use_bias=False,return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(firstLSTMoutput, input_shape=(X.shape[1], X.shape[2]),use_bias=False))

model.add(Dense(X.shape[2], activation='softmax'))
adam_opt=Adam(lr=0.005, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam_opt)
model.summary()

#from keras.models import load_model
#tmp=load_model('20_epoch_2lstm_256').get_weights()
#model.set_weights(tmp)

checkpoint = ModelCheckpoint('best_model', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X,y,epochs=10,batch_size=128,callbacks=callbacks_list)
#model.save('30_epoch_2lstm_256')
