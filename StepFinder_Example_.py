# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:57:04 2021

@author: motorsgroup
"""

#%%
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tensorflow as tf
import warnings

warnings.simplefilter("ignore")

sec=1
Hz=1

#%%
print(tf.__version__)
print(tf.config.list_physical_devices())

#%%
sampling=10000 *(Hz)
samples=2000

dt=1/sampling

#%%
def f(x,xc,amp): return 0.5*(np.sign(x-xc)+1)*amp

#
t=np.linspace(0,samples/sampling,samples) 
y_=np.zeros(t.shape)
t0=0.03 *(sec)
amp=1
while (True):   
    # always incresing step
    # dice_t=np.random.choice(np.arange(0.03,0.10,0.01)) *(sec)
    # y_=y_+f(t,t0,1)
    
    # 
    dice_t=0.03 *(sec)
    y_=y_+f(t,t0,1)
    
    #
    # dice_t=0.05 *(sec)
    # y_=y_+f(t,t0,amp)
    # amp=amp-0.05

    t0=t0+dice_t
    if t0>t[samples-1]:
        break

# add noise
y=y_+np.random.normal(0,1,samples)

# normalize
y_max=np.max(y)
y_min=np.min(y)
y=(y-y_min)/(y_max-y_min)

plt.figure(figsize=(10,5))
plt.plot(t,y,'.')
plt.xlabel('Time [sec]',fontweight='bold')
plt.ylabel('Signal',fontweight='bold')
plt.grid(lw=0.3)
plt.show()

#%%
# Create the model 
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,activation='linear',input_shape=[1]))
model.add(tf.keras.layers.Dense(units=512,activation='tanh'))
model.add(tf.keras.layers.Dense(units=128,activation='tanh'))
model.add(tf.keras.layers.Dense(units=32,activation='tanh'))
model.add(tf.keras.layers.Dense(units=1,activation='linear'))
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Display the model
model.summary()


#%% Train
def step_decay(epoch):
  initial_lrate = 1e-2
  drop = 0.9
  epochs_drop = 20
  lrate = initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
  return lrate

with tf.device('/gpu:0'):
    xdata=t
    ydata=y
    model.fit(xdata,ydata,epochs=1000,verbose=1,callbacks=[tf.keras.callbacks.LearningRateScheduler(step_decay)])
    y_predicted= model.predict(xdata)

# Display the result
y=y*(y_max-y_min)+y_min
y_predicted=y_predicted*(y_max-y_min)+y_min

window_length=667
polyorder=1
# y_dc=savgol_filter(y,window_length,polyorder)

fig,axes=plt.subplots(3,1,figsize=(10,15))
axes[0].plot(t,y,'.',label='Data')
# axes[0].plot(t,y_dc,label='moving average')
axes[0].set_xlabel('Time [sec]',fontweight='bold')
axes[0].set_ylabel('Signal',fontweight='bold')
axes[0].legend()
axes[0].set_ylim((np.floor(np.min(y))-1,np.floor(np.max(y))+1))
axes[0].grid()

axes[1].plot(t,y,'.',label='Data')
axes[1].plot(t,y_predicted,'r',label='NN_StepFinder')
axes[1].set_xlabel('Time [sec]',fontweight='bold')
axes[1].set_ylabel('Signal',fontweight='bold')
axes[1].legend()
axes[1].set_ylim((np.floor(np.min(y))-1,np.floor(np.max(y))+1))
axes[1].grid()

axes[2].plot(t,y_,label='Answer')
axes[2].plot(t,y_predicted,'r',label='NN_StepFinder')
axes[2].set_xlabel('Time [sec]',fontweight='bold')
axes[2].set_ylabel('Signal',fontweight='bold')
axes[2].legend()
axes[2].set_ylim((np.floor(np.min(y))-1,np.floor(np.max(y))+1))
axes[2].grid()

#%% Clear model
tf.keras.backend.clear_session()





