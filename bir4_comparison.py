import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
dt = 2e-6
b1 = np.arange(0.0, 1.0, 0.02)  # gauss, b1 range to sim over

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=128, ptype='ex', flip=np.pi / 2, pbw=0.5,
                        pbc=[0.25], d1e=0.1, d2e=0.1,
                        rampfilt=False, bs_offset=7500)
rfp_bs += rfp_ss

a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxybs = 2 * np.conj(a) * b

[rfp_rfse_am, rfp_rfse_fm] = rf.dz_b1_rf(dt=dt, tb=6, ptype='ex', flip=np.pi/2, pbw=0.5, pbc=0.25,
                                       d1=0.1, d2=0.1, os=8, split_and_reflect=True)

# pl.LinePlot(rfp_rfse_am)
# pl.LinePlot(rfp_rfse_fm)
b1 = np.reshape(b1, (np.size(b1),1))
[a, b] = rf.sim.abrm_nd(2*np.pi*dt*rfp_rfse_fm, b1, 2*np.pi*4258*dt*np.reshape(rfp_rfse_am, (np.size(rfp_rfse_am),1)))
Mxyrfse = -2*np.real(a*b) + 1j*np.imag(np.conj(a)**2 - b**2)

n = np.size(rfp_rfse_am)
dw0 = 100 * np.pi / dt / n
beta = 20
kappa = np.arctan(10)
flip = np.pi / 2
[am_bir, om_bir] = rf.adiabatic.bir4(n, beta, kappa, flip, dw0)

# pl.LinePlot(am_bir)
# pl.LinePlot(om_bir)
# check relatively homogeneous over range of B1 values
b1 = np.reshape(b1, (np.size(b1), 1))
a = np.zeros(np.shape(b1), dtype='complex')
b = np.zeros(np.shape(b1), dtype='complex')

for ii in range(0, np.size(b1)):
    [a[ii], b[ii]] = rf.sim.abrm_nd(
        2 * np.pi * dt * 4258 * b1[ii] * am_bir, np.ones(1),
        dt * np.reshape(om_bir, (np.size(om_bir), 1)))

Mxybir = 2 * np.multiply(np.conj(a), b)


# comparing the two pulses across duration, power, etc.
print('BSSE duration = {} s'.format(dt*np.size(rfp_bs)))
print('RFSE duration = {} s'.format(dt*np.size(rfp_rfse_am)))
print('BIR4 duration = {} s'.format(dt*np.size(am_bir)))


pyplot.figure()
pyplot.plot(b1, np.abs(Mxybs.transpose()), label ='BSSE pulse')
pyplot.plot(b1, np.abs(Mxyrfse.transpose()), label ='RFSE pulse')
pyplot.plot(b1, np.abs(Mxybir), label ='BIR-4 pulse')
pyplot.legend(loc = 'upper left')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.show()