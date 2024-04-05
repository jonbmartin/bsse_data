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
dt = 4e-6

# # CHIRP PULSE
# bsrf, rfp_ex = rf.dz_bssel_chirp_rf(dt=dt, T=0.005448, pbb=1.8, pbt=2.2, bs_offset=20000)
#
# a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
#                     np.array([[1]]), 0, b1.reshape(len(b1), 1))
# MxyBS = 2 * np.conj(a) * b
# a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
#                     np.array([[1]]), 0, b1.reshape(len(b1), 1))
# Mxy90EX = 2 * np.conj(a) * b
# a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
#                     np.array([[1]]), 0, b1.reshape(len(b1), 1))
# MxyEXFULL = 2 * np.conj(a) * b
#
# print('CHIRP pulse dur = {}'.format(dt * np.size(bsrf)))

# axs[0, 0].plot(b1, np.abs(MxyEXFULL.transpose()))
# axs[0, 0].plot(b1, np.abs(MxyBS.transpose()))
# axs[0, 0].plot(b1, np.abs(Mxy90EX.transpose()))
# axs[0,0].set_xlabel('$B_1$ (Gauss)')
# axs[0,0].set_ylabel('$|M_{xy}|$')
# # axs.set_ylabel('|Mxy|')
# # axs.set_xlabel('Gauss')

# EXCITATION PULSE
bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype='st',
                                  flip=np.pi / 2, pbw=0.35, pbc=[2.0],
                                  d1e=0.01, d2e=0.01, rampfilt=False,
                                  bs_offset=10000)

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

axs[0, 1].plot(b1, np.abs(MxyEXFULL), label ='')
axs[0, 1].plot(b1, np.abs(MxyBS), label ='')
axs[0, 1].plot(b1, np.abs(Mxy90EX), label ='')
axs[0,1].set_xlabel('$B_1$ (Gauss)')
axs[0,1].set_ylabel('$|M_{xy}|$')
axs[0,1].legend(('EX + BS','EX only','BS only'),shadow=True)

# MULTI-B1 PULSE
bsrf, rfp_ex, _  = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype='ex',
                                  flip=np.pi / 2, pbw=0.438, pbc=[1.5, 2.25, 3.0],
                                  d1e=0.01, d2e=0.01, rampfilt=False,
                                  bs_offset=15000)

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyBS = 2 * np.conj(a) * b
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90EX = 2 * np.conj(a) * b
a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFULL = 2 * np.conj(a) * b

print('MB pulse dur = {}'.format(dt * np.size(bsrf)))

axs[1,1].plot(b1, np.abs(MxyEXFULL.transpose()), label ='')
axs[1,1].plot(b1, np.abs(MxyBS.transpose()), label ='')
axs[1,1].plot(b1, np.abs(Mxy90EX.transpose()), label ='EX')
axs[1,1].set_xlabel('$B_1$ (Gauss)')
axs[1,1].set_ylabel('$|M_{xy}|$')

# INVERSION PULSE
bsrf, rfp_ex, _  = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype='inv',
                                  flip=np.pi, pbw=0.35, pbc=[2.0],
                                  d1e=0.01, d2e=0.01, rampfilt=False,
                                  bs_offset=15000)

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

axs[2, 1].plot(b1, MzFULL.transpose())
axs[2, 1].plot(b1, MzBS.transpose())
axs[2, 1].plot(b1, MzINV.transpose())
axs[2,1].set_xlabel('$B_1$ (Gauss)')
axs[2,1].set_ylabel('$M_{z}$')

pyplot.show()






