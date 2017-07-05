import pandas
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import math
import scipy
from scipy.stats import pearsonr as r
from glob import glob


bscount = 100
path = "S:/Mark/Research/Fish Behavioural/30062017 Arnold Looming/Run2"

track = ""
profile = "looming.csv"
asrun = "looming.npy"

# Offset in seconds
# startoffset = 2*60+59
startoffset = 6

plt.figure()


def find_nearest(array, value):
    # find closest array entry that is BELOW the search value
    # diff = array-value
    # diff[diff > 0] = -1e6
    # idx = diff.argmax()
    # return idx
    return (np.abs(array-value)).argmin()

def import_asrun(filename):
    file = np.load(filename)
    log_mat = np.zeros((len(file), 11))
    for ix, row in enumerate(file):
        log_mat[ix, :] = np.array(row.split(' '))[[0, 2,3,4,5,6,7,8,9,10,11]]
    return log_mat


def generate_correlated_list():

    global track, profile, asrun, startoffset, corlist

    os.chdir(path)
    track = glob("*track.csv")[0]
    track = pandas.read_csv(track)
    profile = np.genfromtxt(profile, delimiter=',')
    asrun = import_asrun(asrun)
    asrun[:,0] = (asrun[:,0] - asrun[0,0])/1000


    track = track.values
    trackLength = track.shape[0]
    profileLength = profile.shape[0]
    asrunLength = asrun[-1,0]

    # convert timescale to seconds, match the profile
    track[:, 1] = track[:, 1]/1000

    sIn = np.abs(track[:,1]-startoffset).argmin()
    eIn = np.abs(track[:,1]-(startoffset + asrunLength)).argmin()

    if sIn > eIn:
        print("Indexing failure")
        exit()
    tslice = track[sIn:eIn]
    tslice[:,1] = tslice[:,1]-tslice[0,1]

    #at this point I should have tslice and asrun roughtly time aligned

    corlist = np.zeros(shape=(tslice.shape[0], 2))
    for ix, time in enumerate(tslice[:,1]):
        # take nearest smaller time value for each tslice index
        # print(ix, time)
        asIndex = find_nearest(asrun[:,0], time)
        # print(asIndex, asrun[asIndex,0])
        # Independant var: Resistor value
        corlist[ix, 0] = asrun[asIndex, 1]
        # Dependant var: X position
        corlist[ix, 1] = tslice[ix, 2]

    r(corlist[:,0], corlist[:,1])


def bootstrap():

    np.random.shuffle(corlist[:, 1])

    # plt.plot(corlist)
    # plt.figure()
    # output = plt.xcorr(corlist[:, 0], corlist[:, 1], maxlags=274)


def xcorr(a: np.ndarray, b: np.ndarray, maxlags=274):
    """
    xcorr implementation!
    plt.xcorr(x, y, maxlags=phaseshift, normed=True)
    corresponds to
    cor = np.correlate(a, b, mode='full')/(np.std(a) * np.std(b) * len(a))
    note that I've only used this with len(a)=len(b). I'm guessing that it's actually the average length?
    unnormed it's just correlate with mode='full'
    the 'lags' values are just a slice the cor array around (cor.shape[0]-1)/2
    """
    sdA = np.std(a)
    sdB = np.std(b)
    if math.isclose(sdA, 0) or math.isclose(sdB, 0):
        print("Std deviation too small")
        exit()
    cor = np.correlate(a / sdA, b / sdB, mode='full') * 2/(a.shape[0] + b.shape[0])
    ce = int((cor.shape[0]-1)/2)
    return cor[ce - maxlags:ce + maxlags]


# body of code
generate_correlated_list()
corlist[:, 0] = corlist[:, 0] - corlist[:, 0].mean()
corlist[:, 1] = corlist[:, 1] - corlist[:, 1].mean()
output = xcorr(corlist[:, 0], corlist[:, 1], maxlags=274)
plt.plot(np.arange(-274, 274), output)
plt.grid()

# plt.figure()
bootarray = np.zeros((274*2, bscount))
bootarray[:, 0] = np.arange(-274, 274)

for ix in range(1, bscount):
    bootstrap()
    output = xcorr(corlist[:, 0], corlist[:, 1], maxlags=274)
    bootarray[:, ix] = output
    plt.plot(bootarray[:,0], bootarray[:,ix])



plt.show()


