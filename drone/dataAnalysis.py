from mat73 import loadmat
from functools import reduce
from scipy.fft import fft
from scipy.fftpack import fftfreq
from datetime import datetime
import numpy as np

def car(sample, allSamples):
    return sample - allSamples.mean()

def ambientFreq(signal, frequencies, step):
    average = np.zeros(step)
    count = 0

    i = step
    length = len(signal)
    progress = 0
    while i < length:
        if sum(map(lambda f: sum(f), frequencies[i-step:i])) == 0:
            average = np.add(average, np.abs(fft(signal[i-step:i])))
            count += 1
        if i / length > progress:
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            print(f"{date} {int(round(progress * 100))}% done")
            progress += 0.1
        i += step
    print(f"{date} {int(progress * 100)}% done")
    return average / count

def nonZeroFreqTest(signal, frequencies, step):
    i = step + 50000
    length = len(signal)
    progress = 0
    while i < length:
        #print(f"{i} {sum(frequencies[i-step:i, 1])}")
        if sum(frequencies[i-step:i, 1]) >= 0.9 * step:
            return np.abs(fft(signal[i-step:i]))
        if i / length > progress:
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            print(f"{date} {int(round(progress * 100))}% done")
            progress += 0.1
        i += 1
    print(f"{date} {int(progress * 100)}% done")
    return None

if __name__ == "__main__":
    matrix = loadmat('dane.mat')['matrix']
    step = 2000
    frequency = 1000
    allProbes = matrix[:, 5:21]
    frequencies = matrix[:, 26:28]
    signal1 = matrix[:, 11]
    carSignal1 = list(map(car, signal1, allProbes))
    ambientFreqPow1 = ambientFreq(carSignal1, frequencies, step)
    freqValues = fftfreq(step, 1 / frequency)

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

    y1 = ambientFreqPow1[minIndex:maxIndex]
    y2 = nonZeroFreqTest(carSignal1, frequencies, step)[minIndex:maxIndex]
    x = freqValues[minIndex:maxIndex]

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.grid()
    plt.savefig("test")

