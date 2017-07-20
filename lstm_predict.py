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

from keras.models import load_model
model=load_model('20_epoch_2lstm_256')


#seed=dataX_L[5:5+timesteps]

def sample(a, temperature=1.0):
   # helper function to sample an index from a probability array
   a = np.log(a) / temperature
   a = np.exp(a) / np.sum(np.exp(a))
   a=a/sum(a)-1e-5
   #print(sum(a))s
   return np.argmax(np.random.multinomial(1, a, 1))

timesteps=30
dataX_L=[ohe[c2ndic[c]] for c in cha[:300]]

def gen_text(n,seed=np.array(dataX_L[5:5+timesteps]),model=model,t=1):
    text=seed
    print('seed is:',vts(text))
    inp=np.array(text)
    for i in range(n):
        prear=model.predict(np.reshape(inp[-timesteps:],(1,timesteps,len(text[0]))))
        pre=sample(np.reshape(prear,len(prear[0])),temperature=t)
        text=np.concatenate((text,[ohe[pre]]),axis=0)
        inp=np.array(text)
    first_sentence=np.array(text)
    indx=[np.argmax(pr) for pr in first_sentence[timesteps:]]
    a=''
    for i in indx:
        a=a+n2cdic[i]
    return a

def gen_poem(seed=np.array(dataX_L[5:5+timesteps]),model=model,t=1):
    newline=0
    seedlength=len(seed)
    text=np.array(np.concatenate((seed,[ohe[c2ndic['\n']]]),axis=0))
    old=c2ndic['p']
    while(newline<3):
        prear=model.predict(np.reshape(text[-timesteps:],(1,timesteps,len(text[0]))))
        pre=sample(np.reshape(prear,len(prear[0])),temperature=t)
        text=np.concatenate((text,[ohe[pre]]),axis=0)
        old=n2cdic[pre]
        if(t>0.2):
            if(pre==c2ndic['\n'] and not old==c2ndic['\n']):
                newline+=1
            if(pre==c2ndic['\n'] and old==c2ndic['\n']):
                text=text[:-1]
        if(t<=0.2):
            if(pre==c2ndic['\n']):
                newline+=1
    return ''.join([ n2cdic[np.argmax(c)] for c in list(text[seedlength+1:])])
print(gen_poem())

def vts(vec):
    first_sentence=np.array(vec)
    indx=[np.argmax(pr) for pr in first_sentence]
    a=''
    for i in indx:
        a=a+n2cdic[i]
    return a
