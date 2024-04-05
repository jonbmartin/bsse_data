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

figs, axs = pyplot.subplots(3,2)
# problem parameters
b1 = np.arange(0, 4, 0.01)  # gauss, b1 range to sim over
dt = 2e-6
offset = 7500
tb = 4
d1e = 0.01
d2e = 0.01
# small-tip
bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='st',
                                  flip=np.pi / 4, pbw=0.3, pbc=[1.4],
                                  d1e=d1e, d2e=d2e, rampfilt=True,
                                  bs_offset=offset)

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyBS = np.squeeze(2 * np.conj(a) * b)
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90EX = np.squeeze(2 * np.conj(a) * b)
a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFULL = np.squeeze(2 * np.conj(a) * b)

print('MB pulse dur = {}'.format(dt * np.size(bsrf)))

t = np.arange(0,np.size(bsrf)*dt,dt)*1000
axs[0, 0].plot(t, abs(bsrf + rfp_ex).T, 'k')
axs[0, 0].set_xlabel(r't (ms)')
axs[0, 0].set_title(r'|RF| (a.u.)')

axs[0, 1].plot(b1, np.abs(MxyBS), '#2ca02c', label ='')
axs[0, 1].plot(b1, np.abs(Mxy90EX), '#ff7f0e', label ='')
axs[0, 1].plot(b1, np.abs(MxyEXFULL), '#1f77b4', label ='')
axs[0, 1].set_xlabel('$B_1$ (Gauss)')
axs[0, 1].set_ylabel('$|M_{xy}|$')
axs[0, 1].legend(('BS only', 'SS only', 'BS + SS'), shadow=True)
axs[0, 1].set_ylim([0,1.2])


# 90 EXCITATION PULSE
bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex',
                                  flip=np.pi / 2, pbw=0.3, pbc=[2],
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
axs[1, 0].plot(t, abs(bsrf + rfp_ex).T, 'k')
axs[1, 0].set_xlabel(r't (ms)')

axs[1, 1].plot(b1, np.abs(MxyBS), '#2ca02c', label ='')
axs[1, 1].plot(b1, np.abs(Mxy90EX), '#ff7f0e', label ='')
axs[1, 1].plot(b1, np.abs(MxyEXFULL), '#1f77b4', label ='')
axs[1, 1].set_xlabel('$B_1$ (Gauss)')
axs[1, 1].set_ylabel('$|M_{xy}|$')
axs[1, 1].set_ylim([0, 1.2])

# INVERSION PULSE
bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='inv',
                                  flip=np.pi, pbw=0.3, pbc=[2.0],
                                  d1e=d1e, d2e=d2e, rampfilt=False,
                                  bs_offset=offset)

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzBS = 1-2*np.abs(b)**2

a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzINV = 1-2*np.abs(b)**2
a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzFULL = 1-2*np.abs(b)**2

print('INV pulse dur = {}'.format(dt * np.size(bsrf)))

t = np.arange(0, np.size(bsrf)*dt, dt)*1000
axs[2, 0].plot(t, abs(bsrf + rfp_ex).T, 'k')
axs[2, 0].set_xlabel(r't (ms)')

axs[2, 1].plot(b1, MzBS.T, '#2ca02c', label ='')  # green
axs[2, 1].plot(b1, MzINV.T, '#ff7f0e', label ='')  # orange
axs[2, 1].plot(b1, MzFULL.T, '#1f77b4', label ='')  # blue
axs[2, 1].set_xlabel('$B_1$ (Gauss)')
axs[2, 1].set_ylabel('$M_{z}$')
axs[2, 1].set_ylim([-1.2, 1.2])

pyplot.tight_layout()
pyplot.show()






