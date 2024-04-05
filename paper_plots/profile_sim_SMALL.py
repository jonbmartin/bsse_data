#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:24:43 2020

@author: guille
"""
import numpy as np
import sys
sys.path.append(r"/Users/guille/Dropbox/code/sigpy_staging/sigpy-rf")
import sigpy.mri.rf as rf
import matplotlib.pyplot as pyplot

figs, axs = pyplot.subplots(1,2)
# problem parameters
b1 = np.arange(0, 4, 0.01)  # gauss, b1 range to sim over
dt = 2e-6
offset = 7500
tb = 4
d1e = 0.01
d2e = 0.01


# 90 EXCITATION PULSE
bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex',
                                  flip=np.pi / 2, pbw=0.3, pbc=[3],
                                  d1e=d1e, d2e=d2e, rampfilt=True,
                                  bs_offset=offset)

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyBS = np.squeeze(2 * np.conj(a) * b)
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90EX =np.squeeze( 2 * np.conj(a) * b)
a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFULL = np.squeeze(2 * np.conj(a) * b)

print('EX pulse dur = {}'.format(dt * np.size(bsrf)))
t = np.arange(0,np.size(bsrf)*dt,dt)*1000
axs[0].plot(t, abs(bsrf + rfp_ex).T, 'k')
axs[0].set_xlabel(r't (ms)')
axs[0].set_title('|RF| (a.u.)')

axs[1].plot(b1, np.abs(MxyBS), '#2ca02c', label ='')
axs[1].plot(b1, np.abs(Mxy90EX), '#ff7f0e', label ='')
axs[1].plot(b1, np.abs(MxyEXFULL), '#1f77b4', label ='')
axs[1].set_xlabel('$B_1$ (Gauss)')
axs[1].set_title('$|M_{xy}|$')
axs[1].set_ylim([0, 1.2])

pyplot.legend(['Only Bloch-Siegert RF', 'Only Slice-Select RF', 'Both'],loc='upper left')
pyplot.tight_layout()
pyplot.show()






