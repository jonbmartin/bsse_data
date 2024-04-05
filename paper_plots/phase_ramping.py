import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf
import sigpy.plot as pl

figs, axs = pyplot.subplots(1,2)
# problem parameters
b1 = np.arange(9,11, 0.01)  # gauss, b1 range to sim over
dt = 2e-6 # s
offsets = [5000, 7500, 80000]  # Hz
tb = 4
d1e = 0.01
d2e = 0.01
rampfilt = True
pbc = 10
pbw = 0.25


# largest offset
bsrf_3, rfp_ex_3, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex',
                                  flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=rampfilt,
                                  bs_offset=offsets[2])

angle_bs = np.unwrap(np.angle(bsrf_3))
angle_bs =np.pad(np.diff(angle_bs),(0,1),'constant')[0,:]/dt
angle_ex = np.unwrap(np.angle(rfp_ex_3))
angle_ex =np.pad(np.diff(angle_ex),(0,1),'constant')[0,:]/dt
# pl.LinePlot(angle_ex)
T = np.size(bsrf_3)*dt
# pl.LinePlot(angle_bs)
# pl.LinePlot(angle_ex)
# bsrf_3_pr = bsrf_3*np.exp(1j * angle_ex)
# rfp_ex_3_pr = rfp_ex_3*np.exp(1j * angle_bs)




a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_3+bsrf_3), np.zeros(np.size(rfp_ex_3)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy_3 = np.squeeze(2 * np.conj(a) * b)

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_3+bsrf_3), np.zeros(np.size(rfp_ex_3)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy_3_pr = np.squeeze(2 * np.conj(a) * b)

t3 = np.arange(0, np.size(rfp_ex_3),1)*1000*dt
axs[0].plot(t3, abs(rfp_ex_3+bsrf_3).T, '#2ca02c')
axs[0].set_title('$|RF|$ (a.u.)')
axs[0].set_xlabel('t (ms)')


axs[1].plot(b1, abs(Mxy_3), '#2ca02c')
axs[1].plot(b1, abs(Mxy_3_pr), '#ff7f0e')

axs[1].set_title('$|M_{xy}|$')
axs[1].set_xlabel('$B_1$ (G)')

axs[1].legend([
               '$\omega_{bs}$'+'={} kHz \n T={}ms'.format(offsets[2]/1000,np.size(rfp_ex_3)*dt*1000),'phase ramped',],
              loc='upper right')

pyplot.show()
