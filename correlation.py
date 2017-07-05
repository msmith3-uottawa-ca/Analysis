import pandas
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import math
import scipy
from scipy.stats import pearsonr as r

path = "S:/Mark/Research/Fish Behavioural/30062017 Arnold Looming/Run2"

track = "2017-06-30_11-11-47_track.csv"
profile = "looming.csv"
asrun = "looming.npy"

# Offset in seconds
# startoffset = 2*60+59
startoffset = 6



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


def main():

    global track, profile, asrun, startoffset, corlist

    os.chdir(path)

    track = pandas.read_csv(track)
    profile = np.genfromtxt(profile, delimiter=',')
    asrun = import_asrun(asrun)
    asrun[:,0] = (asrun[:,0] - asrun[0,0])/1000


    track = track.values
    # profile = profile.values

    trackLength = track.shape[0]
    profileLength = profile.shape[0]
    asrunLength = asrun[-1,0]

    # convert timescale to seconds, match the profile
    track[:, 1] = track[:, 1]/1000

    # track[:,1] is time
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot((profile[:,0] + 2*60 + 59)*1000, profile[:,1])
    # ax2 = ax1.twinx()
    # ax2.plot(track[:,1], track[:,2], 'r-')
    #
    # for ix, time in enumerate(track[:,1]):
    #     nind = find_nearest(profile[:,0], time % (profile[-1,0] + 0.05))
    #     print(nind, profile[nind, 0], time)
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

    # plt.xcorr(corlist[:,0], corlist[:,1])

def bootstrap():

    np.random.shuffle(corlist[:, 1])

    # plt.plot(corlist)
    # plt.figure()
    output = plt.xcorr(corlist[:, 0], corlist[:, 1], maxlags=274)


main()
corlist[:, 0] = corlist[:, 0] - corlist[:, 0].mean()
corlist[:, 1] = corlist[:, 1] - corlist[:, 1].mean()
output = plt.xcorr(corlist[:, 0], corlist[:, 1], maxlags=274)
bootstrap()

