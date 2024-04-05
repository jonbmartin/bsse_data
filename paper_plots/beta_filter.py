import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

# figs, axs = pyplot.subplots(1, 2)

dt = 4e-6  # s
tb = 8
d1e = 0.01
d2e = 0.01
pbc = 1.5
pbw = 0.6
offset =7500
b1 = np.arange(0.5, 2.5, 0.01)  # gauss, b1 range to sim over
ndes = 128

## inversion experiments

d1_inv, d2_inv = 0.01, 0.01

bsrf_filt, rfp_ex_filt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='inv',
                                  flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=True,
                                  bs_offset=offset)

bsrf_NOfilt, rfp_ex_NOfilt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='inv',
                                  flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=False,
                                  bs_offset=offset)

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_filt*1+bsrf_filt*1), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzFILT = np.squeeze(1-2*np.abs(b)**2)
# pl.LinePlot(b)

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_NOfilt+bsrf_NOfilt), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzNOFILT = np.squeeze(1-2*np.abs(b)**2)

pyplot.figure()
pyplot.plot(b1, np.real(MzNOFILT))
pyplot.plot(b1,np.real(MzFILT))
pyplot.title(r'$M_{z}$, Inversion')
pyplot.xlabel(r'$B_1$ (G)')
pyplot.legend([r'standard SLR $B_N(z)$', r'ramped $B_N(z)$'])
pyplot.ylim([-1.1, 1.1])
pyplot.show()

### 90 degree EXCITATION EXPERIMENTS
bsrf_filt, rfp_ex_filt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='ex',
                                  flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=True,
                                  bs_offset=offset)

bsrf_NOfilt, rfp_ex_NOfilt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='ex',
                                  flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=False,
                                  bs_offset=offset)

print('Pulse duration = {} s'.format(dt * np.size(bsrf_filt)))

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_filt*1+bsrf_filt*1), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFILT = np.squeeze(2 * np.conj(a) * b)
# pl.LinePlot(b)

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_NOfilt+bsrf_NOfilt), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXNOFILT = np.squeeze(2 * np.conj(a) * b)
# plot pulses
print(np.size(rfp_ex_filt))
t = np.arange(0, np.size(rfp_ex_filt)*dt, dt)

# plot magnetization profile
pyplot.figure()
pyplot.plot(b1/10, np.abs(MxyEXNOFILT))
pyplot.plot(b1/10,np.abs(MxyEXFILT))
pyplot.title(r'$|M_{xy}|$, 90$^\circ$ FA')
pyplot.xlabel(r'$B_1^+$ (mT)')
pyplot.legend([r'standard SLR $B_N(z)$', r'ramped $B_N(z)$'])
pyplot.ylim([0, 1.1])
pyplot.show()



### SMALL TIP EXCITATION EXPERIMENTS

bsrf_filt, rfp_ex_filt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='st',
                                  flip=np.pi / 6, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=True,
                                  bs_offset=offset)

bsrf_NOfilt, rfp_ex_NOfilt, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=ndes, ptype='st',
                                  flip=np.pi / 6, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=False,
                                  bs_offset=offset)

t = np.arange(0,dt*np.size(bsrf_filt),dt)*1000
pyplot.plot(t, np.squeeze(abs(bsrf_NOfilt+rfp_ex_NOfilt)), color='k')
pyplot.title(r'BSSE 30$^\circ$ TB=8 pulse')
pyplot.xlabel('time (ms)')
pyplot.ylabel('a.u.')
pyplot.show()
print('Pulse duration = {} s'.format(dt * np.size(bsrf_filt)))

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_filt+bsrf_filt*1), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXFILT = np.squeeze(2 * np.conj(a) * b)

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_NOfilt+bsrf_NOfilt), np.zeros(np.size(rfp_ex_filt)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
MxyEXNOFILT = np.squeeze(2 * np.conj(a) * b)
# plot pulses
print(np.size(rfp_ex_filt))
t = np.arange(0, np.size(rfp_ex_filt)*dt, dt)

# plot magnetization profile
pyplot.figure()
pyplot.plot(b1/10, np.abs(MxyEXNOFILT))
# pyplot.plot(b1/10,np.abs(MxyEXFILT))
pyplot.title(r'BSSE 30$^\circ$ Flip Angle')
pyplot.xlabel(r'$B_1^+$ (mT)')
pyplot.ylabel(r'$|M_{xy}|/M_0$')
pyplot.legend([r'standard SLR $B_N(z)$', r'ramped $B_N(z)$'])
pyplot.ylim([0, 1.1])
pyplot.show()

## inversion experiments REFINEMENT
data_refined = sio.loadmat('after_refine50.mat')
print('refined profile')
mz_ini = data_refined['Mz_ini']
mz_fin = data_refined['Mz_fin']
pulse_init = data_refined['pulse_ini']
pulse_fin = data_refined['pulse_ref']
loss = data_refined['loss']
pyplot.figure()
pyplot.plot(b1/10, np.real(MzNOFILT))
# pyplot.plot(b1,np.real(MzFILT))
pyplot.plot(b1/10, np.squeeze(np.real(mz_fin)))
pyplot.title(r'$M_{z}$, Inversion')
pyplot.xlabel(r'$B_1^+$ (mT)')
pyplot.legend([r'standard SLR $B_N(z)$', r'iteratively refined'])
pyplot.ylim([-1.1, 1.1])
pyplot.show()
