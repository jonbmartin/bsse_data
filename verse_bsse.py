import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 1e-6
db1 = 0.01
pbc = 1.4
b1min = 0
b1max = pbc+1
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over
epsilon = 1e-12

pbw = 0.3 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 1.5
bs_offset = 5000

# initial design of RF pulse
rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)
# disregard the sweeps. We will look @ those later.
padsize = 591
rfp_ss = rfp_ss[:,padsize:np.size(rfp_ss)-padsize]
rfp_bs = rfp_bs[:,padsize:np.size(rfp_bs)-padsize]
full_pulse_before = rfp_bs + rfp_ss
pl.LinePlot(full_pulse_before)
mean_ss = np.mean(abs(rfp_ss))

err = ((1)-abs(rfp_ss+rfp_bs))
a_k = err+0.05
pyplot.plot(abs(rfp_ss.T))
pyplot.plot(a_k.T)
pyplot.legend(['ss RF', 'a_k'])
pyplot.show()

pl.LinePlot(a_k, title='a_k')
t_k = dt / (a_k+epsilon)
pl.LinePlot(t_k, title='t_k')
b1_k = rfp_ss * a_k
pl.LinePlot(b1_k, title='versed b1')
G = np.ones(np.shape(rfp_ss))
g_k = a_k * G
# rfp_ss = np.concatenate(((np.zeros((1,padsize)), b1_k, np.zeros((1,padsize)))),axis=1)
print(f'pulse duration = {np.size(rfp_ss)*dt*1000} ms')
full_pulse = rfp_bs + b1_k

pyplot.plot(abs(full_pulse_before.T))
pyplot.plot(abs(full_pulse.T))
pyplot.legend(['before VERSE', 'after VERSE'])
pyplot.show()