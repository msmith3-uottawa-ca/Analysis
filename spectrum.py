import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
from glob import glob
from scipy import signal
import re

path = r"S:\Mark\Research\Fish Behavioural\Glass\09082017"
os.chdir(path)
# filename = glob("*track.csv")[0]

for filename in glob("**/*track.csv", recursive=True):
    print(filename)
    track = pandas.read_csv(filename)
    track = track.values
    data = track[:, 2]
    m = re.search(r".*?(.+)\\.+track\.csv", filename)
    if(m.group(1)):
        label = m.group(1)
    else:
        label = "unknown"
    print("Label: " + label)
    data = data - data.mean()
    data = data / max(abs(data.max()), abs(data.min()))

    # plt.plot(data)
    # data = np.sin(np.arange(0, 100, 0.1))
    # data = np.random.rand(301) - 0.5
    ps = np.abs(np.fft.fft(data))**2
    ps2 = np.fft.fft(data)
    time_step = 1 / 30
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    # plt.figure()
    # plt.plot(track[:,1]/1000, data)
    plt.figure()
    plt.plot(freqs[idx], ps[idx])
    plt.title(label)
    plt.figure()

    plt.title(label)
    f, t, Sxx = signal.spectrogram(data, 30, noverlap=200)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

