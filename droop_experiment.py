import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 0.5e-6
b1 = np.arange(0,3, 0.01)  # gauss, b1 range to sim over

pbc = 1.4
pbw = 0.3
tb = 4
d1, d2 = 0.01, 0.01
bs_offset= 7608#8155#7608

# pulse designs
am, fm = rf.dz_b1_rf(dt=dt, tb=tb, ptype='st', flip=np.pi / 4, pbw=pbw,
             pbc=pbc, d1=d1, d2=d2, os=8, split_and_reflect=True)
# pl.LinePlot(am)
bsrf = am * np.exp(1j * dt * 2 * np.pi * np.cumsum(fm))

print('rfse dur = {} s'.format(np.size(am)*dt*1000 ))

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)
print('bsse dur = {} s'.format(np.size(rfp_bs)*dt*1000 ))

full_pulse = rfp_bs + rfp_ss

t = np.linspace(0, np.size(bsrf),np.size(bsrf)) * dt * 1000 # time in ms

# apply droop to both pulses
droop0d5db  = np.linspace(1,0.944,np.size(rfp_bs))
droop1d5db  = np.linspace(1,0.841,np.size(rfp_bs))

full_pulse_droop1d5 = full_pulse * droop1d5db
bsrf_droop1d5 = bsrf * droop1d5db
full_pulse_droop0d5 = full_pulse * droop0d5db
bsrf_droop0d5 = bsrf * droop0d5db


# simulation
# no droop
a_rfse, b_rfse = rf.abrm_hp(2*np.pi*4258*dt*bsrf.reshape((1, np.size(bsrf))), np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_rfse_nodroop = 2 * np.conj(a_rfse) * b_rfse

a_bsse, b_bsse = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_bsse_nodroop = 2 * np.conj(a_bsse) * b_bsse

# 0.5dB droop
a_rfse, b_rfse = rf.abrm_hp(2*np.pi*4258*dt*bsrf_droop0d5.reshape((1, np.size(bsrf))), np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_rfse_0d5droop = 2 * np.conj(a_rfse) * b_rfse

a_bsse, b_bsse = rf.abrm_hp(2*np.pi*4258*dt*full_pulse_droop0d5.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_bsse_0d5droop = 2 * np.conj(a_bsse) * b_bsse

# 1.5dB droop
a_rfse, b_rfse = rf.abrm_hp(2*np.pi*4258*dt*bsrf_droop1d5.reshape((1, np.size(bsrf))), np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_rfse_1d5droop = 2 * np.conj(a_rfse) * b_rfse

a_bsse, b_bsse = rf.abrm_hp(2*np.pi*4258*dt*full_pulse_droop1d5.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull_bsse_1d5droop = 2 * np.conj(a_bsse) * b_bsse

## PLOTTING ##
ax1 = pyplot.subplot(221)
pyplot.plot(t, abs(am)*droop1d5db, color='k')
pyplot.plot(t,np.ones(np.size(bsrf))*1, 'r--')
pyplot.plot(t,np.ones(np.size(bsrf))*0.841, 'r--')
pyplot.title('RFSE A(t) (-1.5 dB droop)')
pyplot.xlabel('Time (ms)')
pyplot.ylabel('a.u.')
pyplot.ylim([-0.05, 1.1])
# pyplot.show()

ax3 = pyplot.subplot(223)
pyplot.plot(t, np.abs(full_pulse_droop1d5.T), color='k')
pyplot.plot(t,np.ones(np.size(bsrf))*0.978, 'r--')
pyplot.plot(t,np.ones(np.size(bsrf))*0.86, 'r--')
pyplot.title('BSSE A(t) (-1.5 dB droop)')
pyplot.xlabel('Time (ms)')
pyplot.ylabel('a.u.')
pyplot.ylim([-0.05, 1.1])


ax4 = pyplot.subplot(224)
pyplot.plot(b1/10, np.abs(Mxyfull_bsse_nodroop.transpose()),'k')
pyplot.plot(b1/10, np.abs(Mxyfull_bsse_0d5droop.transpose()), 'k:')
pyplot.plot(b1/10, np.abs(Mxyfull_bsse_1d5droop.transpose()), 'k--')
pyplot.title(r'Magn. Profile, BSSE')
pyplot.ylabel(r'$|M_{xy}|/M_0$')
pyplot.xlabel('$B_1^+$ (mT)')
pyplot.legend(['no droop', '-0.5 dB', '-1.5 dB'],loc='upper left')
pyplot.ylim([-0.05,0.8])
# pyplot.show()

ax2 = pyplot.subplot(222)
pyplot.plot(b1/10, np.abs(Mxyfull_rfse_nodroop.transpose()), 'k')
pyplot.plot(b1/10, np.abs(Mxyfull_rfse_0d5droop.transpose()), 'k:')
pyplot.plot(b1/10, np.abs(Mxyfull_rfse_1d5droop.transpose()), 'k--')
pyplot.title(r'Magn. Profile, RFSE')
pyplot.ylabel(r'$|M_{xy}|/M_0$')
pyplot.xlabel('$B_1^+$ (mT)')
pyplot.legend(['no droop', '-0.5 dB', '-1.5 dB'],loc='upper left')
pyplot.ylim([-0.05,0.8])
pyplot.show()