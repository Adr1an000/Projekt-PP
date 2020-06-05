from mat73 import loadmat
from functools import reduce
from scipy.fft import fft
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

def localAmbientFreq(signal, frequencies, frame, step, totalFrames = 60000):
    i = frame

    average = np.zeros(step)
    count = 0

    totalAverages = totalFrames / step

    while i >= 0 and count < totalAverages:
        if sum2d(frequencies[i-step:i]) == 0:
            average = np.add(average, np.abs(fft(signal[i-step:i])))
            count += 1
        i += step
    
    return average / count

def findFrame(predicate, step, list):
    i = step
    while i < len(list):
        if predicate(list[i-step:i]):
            return i
        i += 1
    return -1

def rangeFFT(signal, frame, step):
    return np.abs(fft(signal[frame-step:frame]))

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
        if sum(frequencies[i-step:i, 1]) >= 0.9 * step:
        #if frequencies[i-1, 1] == 1:
            return np.abs(fft(signal[i-step:i]))
        i += 1
    return None

def plotToFile(filename, freq, ambientFreqValues, activeFreqValues):
    plt.plot(freq, ambientFreqValues, 'r')
    plt.plot(freq, activeFreqValues, 'g')
    plt.grid()
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    matrix = loadmat('dane.mat')['matrix']

    step = 2000 #przedzial czasowy do fft
    start = 5000 #od jakiego pkt w czasie zaczyna sie analiza
    frequency = 1000 #czestotliwosc sygnalu

    allProbes = matrix[start:, 5:21] #wycinek macierzy zawierajcy sygnal z wszystkich elektrod
    frequencies = matrix[start:, 26:28] #wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
    signal1 = matrix[start:, 11] #sygnal z jednej elektrody potylicznej

    carSignal1 = list(map(car, signal1, allProbes))
    ambientFreqPow1 = ambientFreq(carSignal1, frequencies, step)
    freqValues = fftfreq(step, 1 / frequency)

    frameMajority1 = findFrame(lambda range: sum(range[:, 1]) > 0.9 * len(range), step, frequencies)
    frameSingle1 = findFrame(lambda range: range[-1, 1] == 1, step, frequencies)

    N = step
    T = 1.0 / frequency
    minFreq = 8
    maxFreq = 20

    minIndex = 0
    maxIndex = 0
    while freqValues[maxIndex] < maxFreq:
        if (freqValues[minIndex] < minFreq):
            minIndex += 1
        maxIndex += 1

    framesToTest = {"single": frameSingle1}
    probesToTest = {"probe-1": 11, "probe-2": 12}

    for probeName, probeNumber in probesToTest.items():
        signal = matrix[start:, probeNumber]
        carSignal = list(map(car, signal, allProbes))
        ambientFreqPow = ambientFreq(carSignal, frequencies, step)

        for frameName, frame in framesToTest.items():
            plotToFile(
                f"{probeName}_{frameName}_totalAverage.png",
                freqValues[minIndex:maxIndex], 
                ambientFreqPow[minIndex:maxIndex],
                rangeFFT(carSignal, frame, step)[minIndex:maxIndex]
            )
            localAmbientFreqPow = localAmbientFreq(signal, frequencies, frame, step)
            plotToFile(
                f"{probeName}_{frameName}_localAverage.png", 
                freqValues[minIndex:maxIndex], 
                localAmbientFreqPow[minIndex:maxIndex],
                rangeFFT(carSignal, frame, step)[minIndex:maxIndex]
            )

    plotToFile(
        "test.png", 
        freqValues[minIndex:maxIndex], 
        ambientFreqPow1[minIndex:maxIndex],
        nonZeroFreqTest(carSignal1, frequencies, step)[minIndex:maxIndex]
    )

