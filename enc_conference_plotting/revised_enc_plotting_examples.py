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
import sigpy.plot as pl

figs, axs = pyplot.subplots(3,1)
# problem parameters
b1 = np.arange(0, 4, 0.01)  # gauss, b1 range to sim over
dt = 1e-6


# EXCITATION PULSE
bsrf, rfp_ex, _  = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype='st',
                                  flip=np.pi / 2, pbw=0.35, pbc=[2.0],
                                  d1e=0.01, d2e=0.01, rampfilt=False,
                                  bs_offset=15000)
expulse = bsrf+rfp_ex

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf, np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyBS = 2 * np.conj(a) * b
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_ex, np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90EX = 2 * np.conj(a) * b
a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex+bsrf), np.zeros(np.size(rfp_ex)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFULL = 2 * np.conj(a) * b

print('EX pulse dur = {}'.format(dt * np.size(bsrf)))

axs[0].plot(b1, np.abs(MxyEXFULL.transpose()), label ='')
axs[0].plot(b1, np.abs(MxyBS.transpose()), label ='')
axs[0].plot(b1, np.abs(Mxy90EX.transpose()), label ='')
axs[0].set_ylabel('$|M_{xy}|$', rotation=90, labelpad=15)
axs[0].legend(('EX + BS','EX only','BS only'))
axs[0].set_xticklabels([])


# INVERSION PULSE
bsrf, rfp_ex, _  = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype='inv',
                                  flip=np.pi, pbw=0.35, pbc=[2.0],
                                  d1e=0.01, d2e=0.01, rampfilt=False,
                                  bs_offset=15000)
invpulse = bsrf+rfp_ex


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

axs[1].plot(b1, MzFULL.transpose())
axs[1].plot(b1, MzBS.transpose())
axs[1].plot(b1, MzINV.transpose())
axs[1].set_ylabel('$M_{z}$', rotation=90, labelpad=15)
axs[1].set_xticklabels([])
axs[1].legend(('EX + BS','EX only','BS only'))


# REFOCUSING PULSE
pbc = 4
pbw=0.55
rfp_bs, rfp, _ = rf.dz_bssel_rf(dt=2e-6, tb=4, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=15000)

rfp_bs_inv, rfp_inv, _ = rf.dz_bssel_rf(dt=2e-6, tb=4, ndes=128, ptype='se', flip=np.pi, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=15000)
ex = rfp_bs + rfp
inv = rfp_bs_inv+rfp_inv

print('REF pulse dur = {}'.format(np.size(inv)*dt))
# refocus =np.concatenate([rfp_bs+rfp, rfp_bs_inv+rfp_inv,np.zeros((1,2000))],axis=1)
# pl.LinePlot(refocus)

a90, b90 = rf.abrm_hp(2 * np.pi * 4258 * dt * (ex), np.zeros(np.size(ex.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
a180, b180 = rf.abrm_hp(2 * np.pi * 4258 * dt * (inv), np.zeros(np.size(inv.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))

axs[2].plot(b1, np.abs(b180.T**2), 'r', label ='BSSE Re(beta**2)')
axs[2].set_xlabel('$B_1$ (Gauss)')
axs[2].set_ylabel('Refocusing\nefficiency', rotation=90, labelpad=5)
# axs[2].legend(('Re('+r'$\beta^2$'+')','Im('+r'$\beta^2$'+')'))

# pyplot.show()


# plotting the pulses themselves
figs2, axs2 = pyplot.subplots(3,1)

t = np.expand_dims(np.linspace(0,np.size(expulse),np.size(expulse)),0)
axs2[2].plot(t.transpose(),np.abs(invpulse.transpose()), label ='')
axs2[2].set_ylabel('Normalized AM', rotation=90, labelpad=15)
axs2[2].set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])
axs2[2].set_xlabel('t (ms)')
#
# axs2[1].plot(t.transpose(), np.abs(invpulse.transpose()), label='')
# axs2[1].set_ylabel('Normalized AM', rotation=90, labelpad=15)
# # axs2[1].set_xticks([1000, 2000, 3000, 4000])
# axs2[1].set_xticklabels([0, 1, 2, 3, 4, 5, 6])
# axs2[1].set_xlabel('t (ms)')
pyplot.show()
