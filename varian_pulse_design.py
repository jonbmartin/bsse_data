import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 4e-6
b1 = np.arange(0, 1.5, 0.01)  # gauss, b1 range to sim over

pbc = 0.6
pbw = 0.3

rfp_bs, rfp, _ = rf.dz_bssel_rf(dt=dt, tb=1.5, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                        pbc=[pbc], d1e=0.1, d2e=0.01,
                        rampfilt=True, bs_offset=7000)

full_pulse = rfp_bs + rfp
print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.legend(loc = 'upper left')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.show()

pulse_dic = {"rf_bs":rfp_bs, "rf_ex":rfp}
sio.savemat('pypulse_out.mat', pulse_dic)