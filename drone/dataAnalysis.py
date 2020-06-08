from mat73 import loadmat
from functools import reduce
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from datetime import datetime
import numpy as np
import matplotlib
#matplotlib.use('agg')
#import tensorflow as tf
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

def sum2d(array):
    sum = 0
    for row in range (len(array)):
        for col in range(len(array[0])):
            sum = sum + array[row][col]
    return sum

def car(sample, allSamples):
    """
    sample - float reprezentujacy wartosc z jednej elektrody w danym punkcie czasu
    allSamples - lista zawierajaca wartosci wszystkich elektrod w danym punkcie czasu
    """
    return sample - allSamples.mean()

def ambientFreq(signal, frequencies, step):
    """
    wyznacza czestotliwosc tla (gdy nie ma zadnej czestotliwosci)
    signal - lista wartosci danej elektrody w przedziale czasowym
    frequencies - wycinek oryginalnej macierzy informujacy czy pojawily sie jakies czestotliwosci
    step - przedzial czasowy do fft
    """
    average = np.zeros(step)
    count = 0

    i = step
    length = len(signal)
    progress = 0
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    print(f"{date} {round(progress * 100)}% done")
    while i < length:
        if sum2d(frequencies[i-step:i]) == 0:
            average = np.add(average, np.abs(fft(signal[i-step:i])))
            count += 1
        if i / length > progress:
            progress += 0.1
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            print(f"{date} {round(progress * 100)}% done")
        i += step
    return average / count
def nonZeroFreqTest(signal, frequencies, step):
    """
    szuka 2s przedzialu czestotliwosci wypelnionego w 90% jedynkami i oblicza z niego fft
    signal - lista wartosci danej elektrody w przedziale czasowym
    frequencies - wycinek oryginalnej macierzy informujacy czy pojawily sie jakies czestotliwosci
    step - przedzial czasowy do fft
    """
    i = step
    length = len(signal)
    while i < length:
        if sum(frequencies[i-step:i, 1]) >= 0.90 * step:
            return np.abs(fft(signal[i-step:i]))
        i += 1
    return None
def nonZeroFreqTest2(signal, frequencies, step, s):
    """
    szuka 2s przedzialu czestotliwosci wypelnionego w 90% jedynkami i oblicza z niego fft
    signal - lista wartosci danej elektrody w przedziale czasowym
    frequencies - wycinek oryginalnej macierzy informujacy czy pojawily sie jakies czestotliwosci
    step - przedzial czasowy do fft
    s - poczatek listy
    """
    i = s+step
    length = len(signal)

    while i < length:
        #print(sum(frequencies[i-step:i, 1]))
        if sum(frequencies[i-step:i, 1]) >= 0.6 * step:
            return np.abs(fft(signal[i-step:i])), i
        i += step
    return None, None
def normalize( y):
    ynorm = np.zeros(len(y))

    mnoznik = 1.0 / np.max(y)
    for i, x in enumerate(y):
        if x < 0:
            ynorm[i] = 0
        else:
            ynorm[i] = mnoznik * y[i]
    return ynorm

def integral(ynorm, x):
    xmax = np.argmax(ynorm)
    xintmin = xmax - 2
    xintmax = xmax + 2
    sumint = 0
    # liczenie całki od 15.5 do 16.5
    for i in range(xintmin, xintmax):
        sumint += (ynorm[i + 1] + ynorm[i]) * (x[i + 1] - x[i])
    sumint /= 2
    return sumint

def integral_(ynorm, x):
    integrall = np.zeros(len(ynorm)-2)
  # liczenie całki od 15.5 do 16.5
    for i in range(len(integrall)):
        integrall[i] = ((ynorm[i+1] + ynorm[i]) * (x[i+1] - x[i]))/2
    return integrall
"""
def createModel(imputSize):
    model = tf.keras.models.Sequential()
    model.add(Dense(imputSize, activation = 'sigmoid'))
    model.add(Dense(32, activation = 'sigmoid'))
    model.add(Dense(32, activation = 'sigmoid'))
    model.add(Dense(5,  activation = 'softmax'))
    adam = tf.keras.optimizers.Adam(lr= 0.0005)
    model.compile(loss="mean_squared_error", optimizer=adam)
    return model
"""
def frequenceRange(minFreq, maxFreq, freqValues):
    minIndex = 0
    maxIndex = 0
    while freqValues[maxIndex] < maxFreq:
        if (freqValues[minIndex] < minFreq):
            minIndex += 1
        maxIndex += 1
    return (minIndex, maxIndex)
#def createData(inputX, i):

def machineLearning(x1, x2, direction):
    a = 10


if __name__ == "__main__":
    direction = np.zeros(5)
    matrix = loadmat('dane.mat')['matrix']
    #print(matrix)
    step = 2000 #przedzial czasowy do fft
    start = 5000 #od jakiego pkt w czasie zaczyna sie analiza
    frequency = 1000 #czestotliwosc sygnalu
    allProbes = matrix[:, 5:21] #wycinek macierzy zawierajcy sygnal z wszystkich elektrod
    frequencies = matrix[:, 26:28] #wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
    signal1 = matrix[:, 11] #sygnal z jednej elektrody potylicznej
    signal2 = matrix[:, 12] #sygnal z jednej elektrody potylicznej

    carSignal1 = list(map(car, signal1, allProbes))
    carSignal2 = list(map(car, signal2, allProbes))
    T = 1.0 / frequency
    freqValues = fftfreq(step, T)
    N = step

    (minIndex, maxIndex) = frequenceRange(8, 20, freqValues)
    s = start
    x = freqValues[minIndex:maxIndex]
    ambientFreqPow1 = ambientFreq(carSignal1[start:], frequencies[start:], step)
    ambientFreqPow2 = ambientFreq(carSignal2[start:], frequencies[start:], step)
    yBackground1 = ambientFreqPow1[minIndex:maxIndex]
    yBackground2 = ambientFreqPow2[minIndex:maxIndex]
    while s is not None:

        y1, s = nonZeroFreqTest2(carSignal1[s:], frequencies[s:], step, s)
        y2, s = nonZeroFreqTest2(carSignal2[s:], frequencies[s:], step, s)

        if y1 is not None and y2 is not None:
            y1 = y1[minIndex:maxIndex]
            ysr1 = np.subtract(y1, yBackground1)
            ynorm1 = normalize(ysr1)
            plt.plot(x, ynorm1, 'r')

            y2 = y2[minIndex:maxIndex]
            ysr2 = np.subtract(y2, yBackground2)
            ynorm2 = normalize(ysr2)
            plt.plot(x, ynorm2, 'g')

            plt.show()

            direction[0] = frequencies[s, 0]
            direction[1] = frequencies[s, 1]
            int1 = integral_(ynorm1, x)
            int2 = integral_(ynorm2, x)
            machineLearning(int1, int2, direction )




