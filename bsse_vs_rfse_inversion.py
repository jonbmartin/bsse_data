import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
dt = 2e-6
b1 = np.arange(0, 4, 0.01)  # gauss, b1 range to sim over

pbc = 2
pbw = 0.5

rfp_bs, rfp, _ = rf.dz_bssel_rf(dt=2e-6, tb=4, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=20000)

rfp_bs_inv, rfp_inv, _ = rf.dz_bssel_rf(dt=2e-6, tb=4, ndes=128, ptype='se', flip=np.pi, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=20000)
ex = rfp_bs + rfp
inv = rfp_bs_inv+rfp_inv * np.exp(-1j*np.pi/2)
# refocus =np.concatenate([rfp_bs+rfp, rfp_bs_inv+rfp_inv,np.zeros((1,2000))],axis=1)
# pl.LinePlot(refocus)

a90, b90 = rf.abrm_hp(2 * np.pi * 4258 * dt * (ex), np.zeros(np.size(ex.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
a180, b180 = rf.abrm_hp(2 * np.pi * 4258 * dt * (inv), np.zeros(np.size(inv.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))


Mxy_rf_bsse = (2*a90*np.conj(b90))*(-b180**2)
# pl.LinePlot(b180**2, title='beta ** 2')
# pl.LinePlot(Mxy_rf, title='inversion mxy')

[rfse90am, rfse90fm] = rf.dz_b1_rf(dt=2e-6, tb=4, ptype='ex', flip=np.pi/2, pbw=pbw, pbc=pbc,
                                       d1=0.01, d2=0.01, os=8, split_and_reflect=True)

[rfse180am, rfse180fm] = rf.dz_b1_rf(dt=2e-6, tb=4, ptype='se', flip=np.pi, pbw=pbw, pbc=pbc,
                                       d1=0.01, d2=0.01, os=8, split_and_reflect=True)


# pl.LinePlot(rfp_rfse_am)
# pl.LinePlot(rfp_rfse_fm)
b1 = np.reshape(b1, (np.size(b1),1))
[a90rfse, b90rfse] = rf.sim.abrm_nd(2*np.pi*dt*rfse90fm, b1, 2*np.pi*4258*dt*np.reshape(rfse90am, (np.size(rfse90am),1)))
[a180rfse, b180rfse] = rf.sim.abrm_nd(2*np.pi*dt*rfse180fm, b1, 2*np.pi*4258*dt*np.reshape(rfse180am, (np.size(rfse180am),1)))
Mxy_rf_rfse = (2*a90rfse*np.conj(b90rfse))*(-b180rfse**2)
# pl.LinePlot(b180**2, title='beta ** 2')
# pl.LinePlot(Mxy_rf, title='inversion mxy')
pyplot.figure()
pyplot.title('BSSE Beta Squared')
pyplot.plot(b1, np.real(b180.T**2), label ='BSSE Re(beta**2)')
pyplot.plot(b1, np.imag(b180.T**2), label ='BSSE Im(beta**2)')
pyplot.legend(loc = 'upper left')
pyplot.xlabel('Gauss')
pyplot.show()

pyplot.figure()
pyplot.title('RFSE Beta Squared')
pyplot.plot(b1, np.real(b180rfse.T**2), label ='RFSE Re(beta**2)')
pyplot.plot(b1, np.imag(b180rfse.T**2), label ='RFSE Im(beta**2)')
pyplot.legend(loc = 'upper left')
pyplot.xlabel('Gauss')
pyplot.show()

pyplot.figure()
pyplot.title('BSSE Mxy')
pyplot.plot(b1, np.real(1j*b180.T**2), label ='BSSE Mx')
pyplot.plot(b1, np.imag(1j*b180.T**2), label ='BSSE My')
pyplot.legend(loc = 'upper left')
pyplot.xlabel('Gauss')
pyplot.show()

pyplot.figure()
pyplot.title('RFSE Mxy')
pyplot.plot(b1, np.real(1j*b180rfse.T**2), label ='RFSE Mx')
pyplot.plot(b1, np.imag(1j*b180rfse.T**2), label ='RFSE My')
pyplot.legend(loc = 'upper left')
pyplot.xlabel('Gauss')
pyplot.show()
