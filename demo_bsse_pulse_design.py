import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 2e-6
b1 = np.arange(0, 4, 0.02)  # gauss, b1 range to sim over

pbc = 2# b1 (Gauss)
pbw = 0.25 # b1 (Gauss)

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='st', flip=np.pi / 3, pbw=pbw,
                                   pbc=[pbc], d1e=0.01, d2e=0.01,
                                   rampfilt=False, bs_offset=7500)

# pl.LinePlot(rfp_bs)
# pl.LinePlot(rfp_ss)

full_pulse = rfp_bs + rfp_ss/1
# pl.LinePlot(full_pulse)
# comparison_pulse = sio.loadmat('bs_neg_ex_pos.mat')['b1']
# pl.LinePlot(full_pulse-comparison_pulse)
# T = np.size(full_pulse)*dt
#
# full_pulse = rf.bssel_ex_slr(T, dt=1e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/4,
#                  pbw=0.2, pbc=0.4, d1e=0.01, d2e=0.01, rampfilt=True,
#                  bs_offset=50000)
# pl.LinePlot(full_pulse)


print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))

rf_abs = np.abs(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))))
rf_phs = np.angle(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))))
gamgdt = np.zeros(np.size(full_pulse))
xx = np.array([[1]])
dom0dt = 0
b1 = b1.reshape(np.size(b1), 1)

# sio.savemat('simulation_testing.mat',{'rf_abs':rf_abs, 'rf_phs':rf_phs, 'gamgdt':gamgdt,'xx':xx,'b1':b1})

# sio.savemat('bsse_pulse_raw.mat', {'rfp_bs': rfp_bs, 'rfp_ss': rfp_ss})

Mxyfull = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.ylim([0, 1])
pyplot.title('')
pyplot.show()

sio.savemat("bsse_example_pulse.mat", {"ss_pulse":rfp_bs, "bs_pulse":rfp_ss, "b1sim":b1, "mxy":Mxyfull})



