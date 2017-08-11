import pandas
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import math
import scipy
from scipy import stats
from scipy.stats import pearsonr as r
from glob import glob
import re

import tkinter as tk
from tkinter import filedialog


def imshow(array):
    plt.figure()
    plt.imshow(bootstrapPosition[:, 0:2], aspect='auto')
    plt.colorbar()


samples = 5000
trials = 5

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=(("Track Files", "*.csv"),))

m = re.search(r".*/(.+/.+)/.+track\.csv", file_path)
if m and m.group(1):
    label = m.group(1)
else:
    label = "unknown"

track = np.loadtxt(file_path, delimiter=",", skiprows=1)
position = track[:,2]
position = position-position.mean()

# Velocity is "distance per frame"
velocity = np.diff(position)

plt.figure()
plt.title("True Velocity distribution of " + label)
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.hist(velocity, bins=15, normed=True, histtype='bar')
# What do you know, it's normally distributed.
# plt.show()

norm = stats.normaltest(velocity)
print(norm)
print(np.mean(velocity))
print(np.std(velocity))

posmax = position.max()
posmin = position.min()

bootstrapVelocity = np.random.choice(velocity, size=(samples, trials))

# Messy way to deal with the box boundaries
bootstrapPosition = np.ndarray((samples + 1, trials))
bootstrapPosition[0,:] = np.random.choice(position, size=(1, trials))

# Column selection
for ix, bopolist in enumerate(bootstrapPosition[0,:]):
    # Row selection
    for iy in range(1, samples + 1):
        nextvalue = bootstrapPosition[iy-1, ix] + bootstrapVelocity[iy-1, ix]
        if nextvalue > posmax and bootstrapVelocity[iy-1, ix] > 0:
            bootstrapPosition[iy, ix] = bootstrapPosition[iy-1, ix] + bootstrapVelocity[iy-1, ix] * -1
        elif nextvalue < posmin and bootstrapVelocity[iy-1, ix] < 0:
            bootstrapPosition[iy, ix] = bootstrapPosition[iy-1, ix] + bootstrapVelocity[iy-1, ix] * -1
        else:
            bootstrapPosition[iy, ix] = bootstrapPosition[iy-1, ix] + bootstrapVelocity[iy-1, ix]

# Bootstrapped positions before boxing
plt.figure()
plt.title("Bootstrap Velocity Distribution before Boxing")
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.hist(bootstrapVelocity, alpha=0.5, normed=True, histtype='bar', bins=15)

plt.figure()
plt.title("Real and Bootstrapped positions of " + label)
plt.plot(position)
plt.plot(bootstrapPosition)

plt.figure()
plt.title("Position Autocorrelation of " + label)
a = track[:,2]
a = a-a.mean()
plt.xcorr(a,a,maxlags=3000)

for ix in range(0, trials):
    # plt.figure()
    a = bootstrapPosition[:, ix]
    a = a - a.mean()
    plt.xcorr(a, a, maxlags=3000, usevlines=False, linestyle='-', marker=None)
# plt.legend()

bootv = np.diff(bootstrapPosition, axis=0)
plt.figure()
plt.title("Bootstrap Velocity Distribution after Boxing")
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.hist(bootv, alpha=0.5, normed=True, histtype='bar', bins=15)

plt.figure()
plt.xcorr(velocity, velocity, maxlags=3000)
plt.title("Velocity Autocorrelation")

filesave_path = ''
filesave_path = filedialog.asksaveasfilename(filetypes=(("Bootstrap", "*.csv"),), defaultextension=".csv")
if filesave_path:
    np.savetxt(filesave_path, bootstrapPosition, delimiter=",")
