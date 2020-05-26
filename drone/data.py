from mat73 import loadmat
from functools import reduce
from scipy.fft import fft
from scipy.fftpack import fftfreq
import numpy as np

def mapRecordToCas(record):
    casRecord = [0, 0, 0, 0, 0]
    avrg = record[5:21].mean()
    casRecord[0] = record[11] - avrg
    casRecord[1] = record[12] - avrg
    casRecord[2] = record[26]
    casRecord[3] = record[27]
    casRecord[4] = record[26] + record[27]

    return casRecord

class fftData:
    def __init__(self, freq, firstColumnFFT, secondColumnFFT, frequencyPresent):
        self.freq = freq
        self.firstColumnFFT = firstColumnFFT
        self.secondColumnFFT = secondColumnFFT
        self.frequencyPresent = frequencyPresent

def casPieceToFreq(casPiece):
    firstColumn = list(map(lambda c: c[0], casPiece))
    secondColumn = list(map(lambda c: c[1], casPiece))

    freq = fftfreq(len(firstColumn))
    firstColumnFFT = fft(firstColumn)
    secondColumnFFT = fft(secondColumn)

    frequencyPresent = reduce(lambda a, c: a + c[4], casPiece, 0)

    return fftData(freq, firstColumnFFT, secondColumnFFT, frequencyPresent)

if __name__ == "__main__":
    mat = loadmat('dane.mat')

    cas = list(map(mapRecordToCas, mat['matrix']))

    i = 2000
    step = 2000
    averageBackgroundFrequency = fftData(fftfreq(step), np.zeros(step), np.zeros(step), False)

    while i < 2001:
        piece = cas[i-2000:i]
        
        frequencies = casPieceToFreq(piece)
        if not frequencies.frequencyPresent:
            averageBackgroundFrequency.firstColumnFFT = np.add(averageBackgroundFrequency.firstColumnFFT, frequencies.firstColumnFFT)
            averageBackgroundFrequency.secondColumnFFT = np.add(averageBackgroundFrequency.secondColumnFFT, frequencies.secondColumnFFT)
        i += 1
    averageBackgroundFrequency.firstColumnFFT = [abs(x) / step for x in averageBackgroundFrequency.firstColumnFFT]
    averageBackgroundFrequency.secondColumnFFT = [abs(x) / step for x in averageBackgroundFrequency.secondColumnFFT]
    #firstmax = max(averageBackgroundFrequency.firstColumnFFT[1:])
    #print(firstmax)
    #for f in averageBackgroundFrequency.firstColumnFFT[1:]:
    #    print ("*" * int(f / firstmax * 100))



