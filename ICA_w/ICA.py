#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:00:19 2020

@author: Prasanna Bartakke
"""

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
#Fs = 11025

def normalise(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)
    
# Numerically stable sigmoid
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    for alpha in anneal:
        for x in X:
            W += alpha * (np.outer(1 - 2 * sigmoid(np.dot(W, x.T)), x) + np.linalg.inv(W.T))
    return W

def unmix(X, W):
    S = np.zeros(X.shape)
    S = X.dot(W.T)
    return S

print("How many sources? (2 or 4): ")
N = int(input())

if N == 4:
    # Experiment with mixing four source files
    fs1, data1 = wavfile.read('ssm1.wav')
    fs2, data2 = wavfile.read('ssm2.wav')
    fs3, data3 = wavfile.read('sss1.wav')
    fs4, data4 = wavfile.read('sss2.wav')
    Fs = (fs1+fs2+fs3+fs4)//4
    sz = min(data1.size, data2.size, data3.size, data4.size)
    data1 = np.array(data1)
    data1 = np.reshape(data1,(-1,1))
    data2 = np.array(data2)
    data2 = np.reshape(data2,(-1,1))
    data3 = np.array(data3)
    data3 = np.reshape(data3,(-1,1))
    data4 = np.array(data4)
    data4 = np.reshape(data4,(-1,1))
    s = np.concatenate((data1[:sz,:],data2[:sz,:],data3[:sz,:],data4[:sz,:]),axis = 1)
    A = np.array([[0.4,0.3,0.2,0.1],[0.3,0.2,0.1,0.4],[0.2,0.1,0.4,0.3],[0.1,0.4,0.3,0.2]]) #Mixing Matrix
else:
    # Experiment with mixing two source files
    fs1, data1 = wavfile.read('sourceX.wav')
    fs2, data2 = wavfile.read('sourceY.wav')
    Fs = (fs1+fs2)//2
    sz = min(data1.size, data2.size, 200000)
    data1 = np.array(data1)
    data1 = np.reshape(data1,(-1,1))
    data2 = np.array(data2)
    data2 = np.reshape(data2,(-1,1))
    s = np.concatenate((data1[:sz,:],data2[:sz,:]),axis = 1)
    A = np.array([[0.6,0.4],[0.4,0.6]]) #Mixing Matrix

# Main code starts here
s = normalise(s) 
X = np.dot(s,A) # mixing four sound sorces as per specified mixing matrix

# Playing mixed sound signals
for i in range(X.shape[1]):
    print('Playing mixed signal %d' % i)
    play(X[:, i])

W = unmixer(X) # Calling ICA to obtain inv(mixing matrix)
S = normalise(unmix(X, W)) # Unmixing uing obtained W matrix

# Playing the separated sound signals
for i in range(S.shape[1]):
    print('Playing separated signal %d' % i)
    play(S[:, i])

