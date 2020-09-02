from mat73 import loadmat
from functools import reduce
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from datetime import datetime
import numpy as np
import matplotlib
# matplotlib.use('agg')
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import random


def sum2d(array):
    sum = 0
    for row in range(len(array)):
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
        if sum2d(frequencies[i - step:i]) == 0:
            average = np.add(average, np.abs(fft(signal[i - step:i])))
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
        if sum(frequencies[i - step:i, 1]) >= 0.6 * step:
            return np.abs(fft(signal[i - step:i]))
        i += step
    return None


def nonZeroFreqTest2(signal, frequencies, step, s):
    """
    szuka 2s przedzialu czestotliwosci wypelnionego w 90% jedynkami i oblicza z niego fft
    signal - lista wartosci danej elektrody w przedziale czasowym
    frequencies - wycinek oryginalnej macierzy informujacy czy pojawily sie jakies czestotliwosci
    step - przedzial czasowy do fft
    s - poczatek listy
    """
    i = s + step
    length = len(signal)

    while i < length:
        # print(sum(frequencies[i-step:i, 1]))
        if sum(frequencies[i - step:i, 1]) >= 0.6 * step:
            return np.abs(fft(signal[i - step:i])), i
        i += step
    return None, None


def normalize(y):
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
    integrall = np.zeros(len(ynorm) - 2)
    # liczenie całki od 15.5 do 16.5
    for i in range(len(integrall)):
        integrall[i] = ((ynorm[i + 1] + ynorm[i]) * (x[i + 1] - x[i])) / 2
    return integrall


def createModel(imputSize):
    model = tf.keras.models.Sequential()
    model.add(Dense(imputSize, input_dim=imputSize, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(4))
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=adam,
                  metrics=['accuracy'])
    return model


def frequenceRange(minFreq, maxFreq, freqValues):
    minIndex = 0
    maxIndex = 0
    while freqValues[maxIndex] < maxFreq:
        if (freqValues[minIndex] < minFreq):
            minIndex += 1
        maxIndex += 1
    return (minIndex, maxIndex)


def tt(isFrequencies, step, s, freq):
    s += step

    if sum(isFrequencies[s - step:s, 1]) / step < 0.1:
        return 0
    if np.sum(freq == 0) > step * 0.8:
        return 1
    if np.sum(freq == 1) > step * 0.8:
        return 2
    if np.sum(freq == 2) > step * 0.8:
        return 3
    return 0


# def createData(inputX, i):


if __name__ == "__main__":
    matrix = loadmat('dane.mat')['matrix']
    # print(matrix)
    step = 2000  # przedzial czasowy do fft
    start = 5000  # od jakiego pkt w czasie zaczyna sie analiza
    frequency = 1000  # czestotliwosc sygnalu
    allProbes = matrix[:, 5:21]  # wycinek macierzy zawierajcy sygnal z wszystkich elektrod
    frequencies = matrix[:, 26:28]  # wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
    signal1 = matrix[:, 11]  # sygnal z jednej elektrody potylicznej
    signal2 = matrix[:, 12]  # sygnal z jednej elektrody potylicznej
    freq = matrix[:, 21]  # jaka czestotliwosc jest wyswietlana
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
    x_data = list()
    y_data = list()
    while s is not None:

        y1 = nonZeroFreqTest(carSignal1[s:], frequencies[s:], step)
        y2, s = nonZeroFreqTest2(carSignal2[s:], frequencies[s:], step, s)

        if y1 is not None and y2 is not None:
            y1 = y1[minIndex:maxIndex]
            y2 = y2[minIndex:maxIndex]
            y_avg = np.add(y1, y2)
            y_avg /= 2
            # ysr_avg = np.subtract(y_avg, yBackground1)
            ysr_avg = normalize(y_avg)
            # plt.plot(x, ysr_avg, 'r')
            y1 = normalize(y2)
            x_data.append(y1)
            y_data.append(tt(frequencies, step, s, freq[s:s + step]))
            # ysr2 = np.subtract(y2, yBackground2)
            # ynorm2 = normalize(ysr2)
            # plt.plot(x, ynorm2, 'g')

            # plt.show()
            """
            y_avg = np.average(ynorm1, ynorm2)
            x_data.append(np.append(ynorm1, ynorm2, axis=0))
            int1 = integral_(ynorm1, x)
            int2 = integral_(ynorm2, x)
            """
    temp = list()
    for i in range(len(x_data)):
        temp.append([x_data[i], y_data[i]])
    random.shuffle(temp)
    x_data = np.array([i[0] for i in temp])
    y_data = np.array([i[1] for i in temp])
    model = createModel(len(x_data[0]))
    model.fit(x_data[:int(len(x_data) * 0.70)], y_data[:int(len(x_data) * 0.70)], epochs=20000, batch_size=8)
    # model.fit(x_data, y_data, epochs=10000, batch_size=4)
    model.evaluate(x_data[int(len(x_data) * 0.7):], y_data[int(len(x_data) * 0.7):], verbose=2)
    # model.evaluate(x_data, y_data, verbose=2)
    pre = model.predict(x_data)
    pass
