from mat73 import loadmat
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from datetime import datetime
import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
import random

STEP = 2000  # przedzial czasowy do fft
START = 5000  # od jakiego pkt w czasie zaczyna sie analiza
FREQUENCY = 1000  # czestotliwosc sygnalu
T = 1.0 / FREQUENCY


def sum1d(array):
    sum_1d = 0
    for row in range(len(array)):
        sum_1d += array[row]
    return sum_1d


def sum2d(array):
    sum_2d = 0
    for row in range(len(array)):
        for col in range(len(array[0])):
            sum_2d += array[row][col]
    return sum_2d


def car(sample, all_samples):
    """
    sample - float reprezentujacy wartosc z jednej elektrody w danym punkcie czasu
    all_samples - lista zawierajaca wartosci wszystkich elektrod w danym punkcie czasu
    """
    return sample - all_samples.mean()


def ambient_freq(signal, frequencies, step):
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
        if sum1d(frequencies[i - step:i]) == 0:
            average = np.add(average, np.abs(fft(signal[i - step:i])))
            count += 1
        if i / length > progress:
            progress += 0.1
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            print(f"{date} {round(progress * 100)}% done")
        i += step
    return average / count


def non_zero_freq_test(signal, frequencies, step, s):
    """
    szuka 2s przedzialu czestotliwosci wypelnionego w 90% jedynkami i oblicza z niego fft
    signal - lista wartosci danej elektrody w przedziale czasowym
    frequencies - wycinek oryginalnej macierzy informujacy czy pojawily sie jakies czestotliwosci
    step - przedzial czasowy do fft
    s - poczatek listy
    """
    i = s + step if s is not None else step
    length = len(signal)

    while i < length:
        if sum(frequencies[i - step:i]) >= 0.6 * step:
            return np.abs(fft(signal[i - step:i])), i
        i += step
    return None, None


def normalize(y):
    y_norm = np.zeros(len(y))

    factor = 1.0 / np.max(y)
    for i, x in enumerate(y):
        if x < 0:
            y_norm[i] = 0
        else:
            y_norm[i] = factor * y[i]
    return y_norm


def frequency_range(min_freq, max_freq, freq_values):
    min_index = 0
    max_index = 0
    while freq_values[max_index] < max_freq:
        if freq_values[min_index] < min_freq:
            min_index += 1
        max_index += 1
    return min_index, max_index


def tt(step, s, freq, is_freq):
    s += step
    if np.sum(is_freq == 0) > step * .7:
        return 0
    if np.sum(freq == 0) > step * .5:
        return 1
    if np.sum(freq == 1) > step * .5:
        return 2
    if np.sum(freq == 2) > step * .5:
        return 3
    return 0


def prepare_data():
    matrix = loadmat('dane.mat')['matrix']
    all_probes = matrix[:, 5:21]  # wycinek macierzy zawierajcy sygnal z wszystkich elektrod
    frequencies = matrix[:, 27]  # wycinek maciery zawierajacy informacje czy wystapily czestotliwosci
    signal1 = matrix[:, 11]  # sygnal z jednej elektrody potylicznej
    signal2 = matrix[:, 12]  # sygnal z jednej elektrody potylicznej
    freq = matrix[:, 21]  # jaka czestotliwosc jest wyswietlana
    car_signal1 = list(map(car, signal1, all_probes))
    car_signal2 = list(map(car, signal2, all_probes))
    freq_values = fftfreq(STEP, T)
    (minIndex, maxIndex) = frequency_range(8, 20, freq_values)
    x = freq_values[minIndex:maxIndex]
    s = START
    ambient_freq_pow1 = ambient_freq(car_signal1[START:], frequencies[START:], STEP)
    ambient_freq_pow2 = ambient_freq(car_signal2[START:], frequencies[START:], STEP)
    y_background1 = ambient_freq_pow1[minIndex:maxIndex]
    y_background2 = ambient_freq_pow2[minIndex:maxIndex]
    y_avg_background = (y_background1 + y_background2) / 2
    x_data = list()
    y_data = list()
    while s is not None:

        y1, s = non_zero_freq_test(car_signal1[s:], frequencies[s:], STEP, s)
        y2, _ = non_zero_freq_test(car_signal2[s:], frequencies[s:], STEP, s)

        if y1 is not None and y2 is not None:
            y1 = y1[minIndex:maxIndex]
            y2 = y2[minIndex:maxIndex]
            y_avg = np.add(y1, y2)
            y_avg /= 2
            ysr_avg = np.subtract(y_avg, y_avg_background)
            ysr_avg = normalize(ysr_avg)
            plt.plot(x, ysr_avg)
            #plt.show()
            x_data.append(ysr_avg)
            y_data.append(tt(STEP, s, freq[s:s + STEP], frequencies[s:s + STEP]))
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Wyrównanie liczby danych
    counted_y = [sum(y_data == 0),
                 sum(y_data == 1),
                 sum(y_data == 2),
                 sum(y_data == 3)]
    more_than_min_y = [sum(y_data == 0) - min(counted_y) / 2,
                       sum(y_data == 1) - min(counted_y) / 2,
                       sum(y_data == 2) - min(counted_y) / 2,
                       sum(y_data == 3) - min(counted_y) / 2]

    for i, lst in enumerate(more_than_min_y):
        for j in range(int(lst)):
            idx = np.where(y_data == i)[0][0]
            y_data = np.delete(y_data, idx)
            x_data = np.delete(x_data, idx, axis=0)

    return x_data, y_data


def shuffle_data(x_data, y_data):
    temp = list(zip(x_data,y_data))
    random.shuffle(temp)
    x, y= zip(*temp)
    return np.array(x), np.array(y)
    # indices = np.arange(x_data.shape[0])
    # np.random.shuffle(indices)
    # return x_data[indices], y_data[indices]


def save_data(x_data, y_data, path='data/'):
    np.save(path + 'x_data', x_data)
    np.save(path + 'y_data', y_data)


def load_data(path='data/'):
    if os.path.isfile(path + 'x_data.npy') and os.path.isfile(path + 'x_data.npy'):
        x_data = np.load(path + 'x_data.npy')
        y_data = np.load(path + 'y_data.npy')
        print('Loading data complete')
        return x_data, y_data
    return None, None


def create_model(input_size, dropout=0.1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=input_size, input_dim=input_size, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(48, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(12, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(4))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model


def save_model(model, path='saved_model/'):
    model.save(path + 'model')


def load_model(path='saved_model/'):
    return tf.keras.models.load_model(path + 'model')


def main():
    # Prepare data
    #x_data, y_data = load_data()
    # if x_data is None or y_data is None:
    x_data, y_data = prepare_data()
    x_data, y_data = shuffle_data(x_data, y_data)
    save_data(x_data, y_data)

    # Split data
    x_train, y_train = x_data[: int(len(x_data) * .7)], y_data[: int(len(y_data) * .7)]
    x_valid, y_valid = x_data[int(len(x_data) * .7):], y_data[int(len(y_data) * .7):]
    x_test, y_test = x_data[-10:], y_data[-10:]

    # Create, train and save NN model
    model = create_model(len(x_data[0]), dropout=0.4)
    model.fit(x_train, y_train, epochs=15000, batch_size=1)
    save_model(model)
    # Or load model
    # model = load_model()
    # Test model
    model.evaluate(x_valid, y_valid, verbose=1)
    pre = model.predict(x_test)
    for i, item in enumerate(pre):
        print('Test', i, ':', 'predicted:', np.argmax(item), 'expected:', y_test[i])


if __name__ == "__main__":
    main()
