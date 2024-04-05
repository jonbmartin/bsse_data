import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 1e-6
b1 = np.arange(0,9, 0.01)  # gauss, b1 range to sim over

pbc = 5  # b1 (Gauss)
pbw = 9.5 # b1 (Gauss)

rf_wurst = rf.dz_bssel_rf_wurst(dt=2e-6, bs_offset=20000)


full_pulse = rf_wurst
pl.LinePlot(full_pulse)
# comparison_pulse = sio.loadmat('bs_neg_ex_pos.mat')['b1']
# pl.LinePlot(full_pulse-comparison_pulse)
# T = np.size(full_pulse)*dt
#
# full_pulse = rf.bssel_ex_slr(T, dt=1e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/4,
#                  pbw=0.2, pbc=0.4, d1e=0.01, d2e=0.01, rampfilt=True,
#                  bs_offset=50000)
# pl.LinePlot(full_pulse)


print('Pulse duration = {}'.format(np.size(full_pulse)*dt))
dom0dt = np.arange(-450,450,1)*np.pi/180  # radians

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.ylim([0,1])
pyplot.show()

