import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
dt = 2e-6
b1 = np.arange(0, 5, 0.01)  # gauss, b1 range to sim over

rfp_bs = rf.dz_bssel_chirp_rf(dt=2e-6,T=0.005, flip=np.pi/3, pbb=0, pbt=2, bs_offset=15000)

# pl.LinePlot(rfp_bs)

a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxybs = 2 * np.conj(a) * b
#
# [rfp_rfse_am, rfp_rfse_fm] = rf.dz_b1_rf(dt=2e-6, tb=4, ptype='ex', flip=np.pi/2, pbw=0.25, pbc=2,
#                                        d1=0.01, d2=0.01, os=8, split_and_reflect=True)

# # pl.LinePlot(rfp_rfse_am)
# # pl.LinePlot(rfp_rfse_fm)
# b1 = np.reshape(b1, (np.size(b1),1))
# [a, b] = rf.sim.abrm_nd(2*np.pi*dt*rfp_rfse_fm, b1, 2*np.pi*4258*dt*np.reshape(rfp_rfse_am, (np.size(rfp_rfse_am),1)))
# Mxyrfse = -2*np.real(a*b) + 1j*np.imag(np.conj(a)**2 - b**2)


# comparing the two pulses across duration, power, etc.
print('BS duration = {} s'.format(dt*np.size(rfp_bs)))
# print('RFSE duration = {} s'.format(dt*np.size(rfp_rfse_am)))

# SAR
SAR_bs = np.sum(abs(rfp_bs)**2)/np.size(rfp_bs)
# SAR_rfse = np.sum(abs(rfp_rfse_am)**2)/np.size(rfp_rfse_am)
# print('Relative BS average power compared to RFSE: {}'.format(SAR_bs/SAR_rfse))

pyplot.figure()
pyplot.plot(b1, np.abs(Mxybs.transpose()), label ='BS pulse')
# pyplot.plot(b1, np.abs(Mxyrfse.transpose()), label ='RFSE pulse')
pyplot.legend(loc = 'upper left')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.show()