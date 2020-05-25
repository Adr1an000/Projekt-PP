from mat73 import loadmat
from functools import reduce
from scipy.fft import fft
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


def casPieceToFreq(casPiece):
    firstColumn = list(map(lambda c: c[0], casPiece))
    secondColumn = list(map(lambda c: c[1], casPiece))

    freq = fft.freq(len(firstColumn))
    firstColumnFFT = fft(firstColumn)
    secondColumnFFT = fft(secondColumn)

    isFrequencyPresent = reduce(lambda a, c: a + c[4], casPiece)

    return freq, firstColumnFFT, secondColumnFFT, isFrequencyPresent


def filterCasPiecesZero(casPice):
    return reduce(lambda a, r: a + r[4], casPice) == 0

if __name__ == "__main__":
    mat = loadmat('dane.mat')

    cas = list(map(mapRecordToCas, mat['matrix']))
    print(type(cas))
    print(type(cas[0]))
    print(cas[0:10])

    i = 2000
    step = 2000

    casPieces = []
    while i < len(cas):
        casPieces.append(cas[i-2000:i])
        i += 1

    map

    zeroPieces = list(filter(filterCasPiecesZero, casPieces))

