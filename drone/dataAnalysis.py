from mat73 import loadmat
from functools import reduce
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('agg')
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

if __name__ == "__main__":
    matrix = loadmat('dane.mat')['matrix']
    #print(matrix)
    step = 2000 #przedzial czasowy do fft
    start = 5000 #od jakiego pkt w czasie zaczyna sie analiza
    frequency = 1000 #czestotliwosc sygnalu
    allProbes = matrix[:, 5:21] #wycinek macierzy zawierajcy sygnal z wszystkich elektrod
    frequencies = matrix[:, 26:28] #wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
    signal1 = matrix[:, 12] #sygnal z jednej elektrody potylicznej

    carSignal1 = list(map(car, signal1, allProbes))
    ##  ambientFreqPow1 = ambientFreq(carSignal1[start:], frequencies[start:], step)
    ambientFreqPow1 = ambientFreq(carSignal1[start:], frequencies[start:], step)
    T = 1.0 / frequency
    freqValues = fftfreq(step, T)

    N = step

    minFreq = 8
    maxFreq = 20

    minIndex = 0
    maxIndex = 0
    while freqValues[maxIndex] < maxFreq:
        if (freqValues[minIndex] < minFreq):
            minIndex += 1
        maxIndex += 1

    y1 = ambientFreqPow1[minIndex:maxIndex]
    y2 = nonZeroFreqTest(carSignal1[start:], frequencies[start:], step)[minIndex:maxIndex]
    x = freqValues[minIndex:maxIndex]
    ysr = np.subtract(y2, y1)
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.plot(x, ysr, 'b')
    xmax =  np.argmax(ysr)
    xintmin  = xmax -1
    xintmax  = xmax +1
    sumint =0
    #liczenie całki od 15.5 do 16.5
    for i in range(xintmin, xintmax):
        sumint += (ysr[i+1]+ysr[i])*(x[i+1]-x[i])
    sumint/=2
    #print(sumint)
    #normalizacja wykresu, danych
    ynorm = np.zeros(len(ysr))
    mnoznik = 1.0/np.max(ysr)
    #print(np.max(ysr))
    for i in range(xintmin, xintmax+1):
        ynorm[i] = mnoznik*ysr[i]
    #plt.plot(x, ynorm, 'b')
    print(ynorm)


    plt.grid()
    plt.savefig("test")

