#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:24:43 2020

@author: guille
"""
import numpy as np
import pickle
# from scipy.io import loadmat
import sys
sys.path.append(r"/Users/guille/Dropbox/code/sigpy_staging/sigpy-rf")
import sigpy as sp
import sigpy.mri as mr
import sigpy.mri.rf as rf
import sigpy.plot as pl
# import scipy.signal as signal
import matplotlib.pyplot as pyplot
from scipy.interpolate import interp1d
import scipy.io as spio

# problem parameters
bs_dur = 0.0064 # seconds, duration of BS pulse
ss_freq = 400 # Hz, frequency of slice-selective pulse
tb = 1.5
ex_flip = np.pi / 2 # flip angle of slice-selective pulse
dt = 2e-6 # seconds, dwell time
b1 = np.arange(0, 2, 0.01) # gauss, b1 range to sim over

# another strategy: design a BIR pulse and monkey with its middle parts
bs_offset = 20000 
N_end = int(1000 * 1e-6 / 2e-6)
K = 3
am, om = rf.bir4(2 * N_end, K, np.arctan(K), 0, 1 * np.pi * bs_offset)
am = np.real(am)
bs_am = np.zeros(N_end)
bs_fm = np.zeros(N_end)
bs_am[:N_end // 2] = -am[N_end // 2:N_end]
bs_fm[:N_end // 2] = -om[N_end // 2:N_end] / 2 / np.pi + bs_offset
bs_am[N_end:N_end // 2 - 1:-1] = -am[N_end // 2:N_end]
bs_fm[N_end:N_end // 2 - 1:-1] = -om[N_end // 2:N_end] / 2 / np.pi + bs_offset

#bs_fm = rf.bloch_siegert_fm(250, 0.25e-3, b1p=60, k = 42)
bs_cplx = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))
a, b = rf.abrm_hp(2*np.pi*4258*dt*bs_cplx.reshape((1, N_end)), np.zeros(N_end), 
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxySweeps = 2 * np.conj(a) * b

x1 = bs_fm[:N_end // 2]
x2 = bs_fm[N_end // 2:]
bs_fm = np.concatenate((x1, bs_offset*np.ones(int(bs_dur // dt)), x2))
x1 = bs_am[:N_end // 2]
x2 = bs_am[N_end // 2:]
bs_am = np.concatenate((x1 / x1.max(), np.ones(int(bs_dur // dt)), x2 / x2.max()))

# bs_am = np.ones(bs_fm.size, dtype=float)

# simulate the BS pulse, alone, versus b1
# a, b = rf.abrm_hp(2*np.pi*4258*dt*bs_am.reshape((1, len(bs_am))), dt*bs_fm, 
#                         np.array([[1]]), 0, b1.reshape(len(b1), 1))
# Mxy = 2 * np.conj(a) * b
# i have verified that the below gives the same as above
bs_cplx = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))
a, b = rf.abrm_hp(2*np.pi*4258*dt*bs_cplx.reshape((1, len(bs_am))), np.zeros(len(bs_am)), 
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy = 2 * np.conj(a) * b

# what range of frequencies are we looking at during the middle of the BS pulse?
fbs = (4248 * b1)**2 / (2 * bs_offset)
pyplot.figure()
pyplot.plot(b1, fbs)
pyplot.xlabel('Gauss')
pyplot.ylabel('Hertz')
pyplot.title('Frequency offset versus B1 amplitude produced by BS pulse')

# now let's add an excitation pulse to it
rfp = rf.dzrf(256, tb, 'ex', 'ls')
# interpolate to same dwell time as bs_dur
rfp = interp1d(np.linspace(-bs_dur / 2, bs_dur / 2, 256), rfp, kind='cubic')
trf = np.linspace(-bs_dur / 2, bs_dur / 2, int(bs_dur / dt))
rfp = rfp(trf)
# scale for 90 degree flip
rfp = rfp / np.sum(rfp) * ex_flip / (2 * np.pi * 4258 * dt)
# shift to middle of excitation frequency band
rfp = rfp * np.exp(-1j * 2 * np.pi * ss_freq * np.arange( - bs_dur / dt / 2, bs_dur / dt / 2 - 1, 1) * dt)
#rfp = rfp * 2 * np.cos(2 * np.pi * np.max(fbs) / 2 * np.arange( - bs_dur / dt / 2, bs_dur / dt / 2 - 1, 1) * dt)

# simulate this pulse alone, on a uniform frequency axis
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp, 2 * np.pi * dt * np.ones((len(rfp))), 
                  np.arange(0, np.max(fbs), 1))
Mxyrf = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(np.arange(0, np.max(fbs), 1), np.abs(Mxyrf.transpose()))
pyplot.title('RF slice profile versus frequency')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Hz')

# now add them and simulate together
rfp = np.concatenate((np.zeros(len(x1)), rfp, np.zeros(len(x1))))
# scale up rfp to compensate b1 amplitude in middle of slice
rfscale = np.interp(ss_freq, fbs, b1)
pl.LinePlot(bs_cplx + rfp / rfscale, title='Full pulse')

a, b = rf.abrm_hp(2*np.pi*4258*dt*(bs_cplx + rfp / rfscale).reshape((1, len(bs_am))), np.zeros(len(bs_am)), 
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxyfull = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()), label = 'Both pulses together')
pyplot.plot(b1, np.abs(Mxy).transpose(), label = 'BS pulse alone')
pyplot.plot(b1, np.abs(MxySweeps).transpose(), label = 'BS pulse sweeps alone')
pyplot.legend(loc = 'upper left')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')

# do a simulation across b1 + b0
domdt = np.arange(-500, 500, 10) * 2 * np.pi * dt
b1 = np.arange(0, 2, 0.05) # gauss, b1 range to sim over
domdtMat, b1Mat = np.meshgrid(domdt, b1)
a, b = rf.abrm_hp(2*np.pi*4258*dt*(bs_cplx + rfp / rfscale).reshape((1, len(bs_am))), np.zeros(len(bs_am)), 
                    np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
MxyMat = 2 * np.conj(a) * b
MxyMat = MxyMat.reshape((len(b1), len(domdt)))
pyplot.figure()
pl.ImagePlot(MxyMat)

# save the dictionary back out
out = {}
out['bs_cplx'] = bs_cplx
out['rfp'] = rfp
out['totalrf'] = bs_cplx + rfp / rfscale
spio.savemat('bs_ex.mat', out)

