import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf
import sigpy.plot as pl

figs, axs = pyplot.subplots(1,2,figsize=(12,5))
# problem parameters
b1 = np.arange(0, 2.75, 0.01)  # gauss, b1 range to sim over
dt = 2e-6
offsets = [2500, 5000, 10000]  # Hz
tb = 4
d1e = 0.01
d2e = 0.01
rampfilt = True
pbc = 1.4
pbw = 0.3
ptype ='ex'
flip =np.pi/2


# smallest offset
bsrf_1, rfp_ex_1, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype=ptype,
                                  flip=flip, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=rampfilt,
                                  bs_offset=offsets[0])

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_1+bsrf_1), np.zeros(np.size(rfp_ex_1)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy_1 = np.squeeze(2 * np.conj(a) * b)

# medium offset
bsrf_2, rfp_ex_2, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype=ptype,
                                  flip=flip, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=rampfilt,
                                  bs_offset=offsets[1])

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_2+bsrf_2), np.zeros(np.size(rfp_ex_2)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy_2 = np.squeeze(2 * np.conj(a) * b)

# largest offset
bsrf_3, rfp_ex_3, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype=ptype,
                                  flip=flip, pbw=pbw, pbc=[pbc],
                                  d1e=d1e, d2e=d2e, rampfilt=rampfilt,
                                  bs_offset=offsets[2])

a, b = rf.abrm_hp(2*np.pi*4258*dt*(rfp_ex_3+bsrf_3), np.zeros(np.size(rfp_ex_3)),
                    np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy_3 = np.squeeze(2 * np.conj(a) * b)

# duration graph
all_offsets = np.linspace(1000, 15000,20)
pbc_v = [0.7, 1.4, 2.8]
dur_m = np.zeros((len(pbc_v), np.size(all_offsets)))
for jj in range(len(pbc_v)):
    for ii in range(np.size(all_offsets)):
        bsrf_ex, _, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=128, ptype=ptype,
                                             flip=flip, pbw=pbw, pbc=[pbc_v[jj]],
                                             d1e=d1e, d2e=d2e, rampfilt=rampfilt,
                                             bs_offset=all_offsets[ii])
        dur_m[jj, ii] = np.size(bsrf_ex)*dt*1000 # record ms
axs[0].plot(all_offsets/1000, dur_m[0,:].T,'k:')
axs[0].plot(all_offsets/1000, dur_m[1,:].T,'k')
axs[0].plot(all_offsets/1000, dur_m[2,:].T,'k--')
axs[0].vlines(x=[2.5], ymin=[0], ymax=[4.58], color=u'#1f77b4')
axs[0].vlines(x=[5.0], ymin=[0], ymax=[5.278], color=u'#ff7f0e')
axs[0].vlines(x=[10.0], ymin=[0], ymax=[7.304], color=u'#2ca02c')
axs[0].set_ylim([0,18])
axs[0].plot(2.5, 4.58, marker="o", markersize=5,markerfacecolor=u'#1f77b4')
axs[0].plot(5.0, 5.278, marker="o", markersize=5,markerfacecolor=u'#ff7f0e')
axs[0].plot(10, 7.304, marker="o", markersize=5,markerfacecolor=u'#2ca02c')
axs[0].legend(['PBC=0.07 mT', 'PBC=0.14 mT', 'PBC=0.28 mT'])
axs[0].set_ylabel('T (ms)')
axs[0].set_title('Pulse Durations (TB=4)')
axs[0].set_xlabel(r'$\omega_{off}$ (kHz)')
axs[0].set_xticks([2.5, 5, 7.5, 10, 12.5, 15])


# t1 = np.arange(0, np.size(rfp_ex_1),1)*1000*dt
# t2 = np.arange(0, np.size(rfp_ex_2),1)*1000*dt
# t3 = np.arange(0, np.size(rfp_ex_3),1)*1000*dt
# axs[1].plot(t3, abs(rfp_ex_3+bsrf_3).T, '#2ca02c')
# axs[1].plot(t2, abs(rfp_ex_2+bsrf_2).T, '#ff7f0e')
# axs[1].plot(t1, abs(rfp_ex_1+bsrf_1).T, '#1f77b4')
# axs[1].set_title('$|RF|$ (a.u.)')
# axs[1].set_xlabel('t (ms)')


axs[1].plot(b1/10, abs(Mxy_1), '#1f77b4')
axs[1].plot(b1/10, abs(Mxy_2), '#ff7f0e')
axs[1].plot(b1/10, abs(Mxy_3), '#2ca02c')
axs[1].set_title('$|M_{xy}|/M_0$')
axs[1].set_xlabel('$B_1^+$ (mT)')

axs[1].legend(['$\omega_{off}$'+'={} kHz \n T={:.2f}ms'.format(offsets[0]/1000,np.size(rfp_ex_1)*dt*1000),
               '$\omega_{off}$'+'={} kHz \n T={:.2f}ms'.format(offsets[1]/1000,np.size(rfp_ex_2)*dt*1000),
               '$\omega_{off}$'+'={} kHz \n T={:.2f}ms'.format(offsets[2]/1000,np.size(rfp_ex_3)*dt*1000)],
              loc='upper right')

pyplot.show()
