from mat73 import loadmat
import numpy.fft as fft
from datetime import datetime
import numpy as np
from drone.brain import normalize, frequency_range


class DataSource:
    def __init__(self):
        self.data = loadmat('dane.mat')['matrix']
        self.timePoint = 0
        self.STEP = 2000
        self.START = 5000  # od jakiego pkt w czasie zaczyna sie analiza
        self.FREQUENCY = 1000  # czestotliwosc sygnalu
        self.T = 1.0 / self.FREQUENCY

    @staticmethod
    def car(sample, all_samples):
        """
      sample - float reprezentujacy wartosc z jednej elektrody w danym punkcie czasu
      all_samples - lista zawierajaca wartosci wszystkich elektrod w danym punkcie czasu
      """
        return sample - all_samples.mean()

    def readData(self):
        all_probes = self.data[self.timePoint:self.timePoint + self.STEP,
                     5:21]  # wycinek macierzy zawierajcy sygnal z wszystkich elektrod
        frequencies = self.data[self.timePoint:self.timePoint + self.STEP,
                      27]  # wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
        signal1 = self.data[self.timePoint:self.timePoint + self.STEP, 11]  # sygnal z jednej elektrody potylicznej
        signal2 = self.data[self.timePoint:self.timePoint + self.STEP, 12]  # sygnal z jednej elektrody potylicznej
        freq = self.data[self.timePoint:self.timePoint + self.STEP, 21]  # jaka czestotliwosc jest wyswietlana
        car_signal1 = list(map(DataSource.car, signal1, all_probes))
        car_signal2 = list(map(DataSource.car, signal2, all_probes))
        sig1 = np.abs(fft.fft(car_signal1))
        sig2 = np.abs(fft.fft(car_signal2))
        freq_values = fft.fftfreq(self.STEP, self.T)
        (minIndex, maxIndex) = frequency_range(8, 20, freq_values)
        y1 = normalize(sig1[minIndex:maxIndex])
        y2 = normalize(sig2[minIndex:maxIndex])
        self.timePoint += self.STEP
        if self.timePoint >= len(self.data[:, 0]):
            self.timePoint = 0
        temp = list()
        temp.append(np.append(y1.ravel(), y2.ravel()))
        return np.array(temp)
