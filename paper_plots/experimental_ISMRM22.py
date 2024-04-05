import scipy.io as sio
import matplotlib.pyplot as plt
import sigpy.mri.rf as rf
import numpy as np
import sigpy.plot as pl

data = sio.loadmat('final_data.mat')

b1_ax = data['b1_ax']
base_d = data['base_int']
fa_d = data['fa_int']
shift_d = data['shift_int']
tb_d = data['tb_int']
# constant across design
pbw = 0.3
d1 = 0.01
d2 = 0.01
dt = 1e-6
bs_offset =7500


# base pulse
tb=4
pbc=1.4

rfp_bs_base, rfp_ss_base, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_base = rfp_bs_base+rfp_ss_base
t = np.linspace(0,dt*np.size(rfp_base)*1000, np.size(rfp_base))
# pl.LinePlot(abs(rfp_base), title='Base Pulse')
plt.plot(t, abs(rfp_base.T), 'k')
plt.title('Base BSSE Pulse')
plt.xlabel('t (ms)')
plt.ylabel('a.u.')
plt.show()

a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_base.reshape((1, np.size(rfp_base))), np.zeros(np.size(rfp_base)),
                    np.array([[1]]), 0, b1_ax.reshape(np.size(b1_ax), 1))
Mxybase = 2 * np.conj(a) * b

# FA pulse
tb=4
pbc=1.4
rfp_bs_fa, rfp_ss_fa, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_fa = rfp_bs_fa+rfp_ss_fa
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_fa.reshape((1, np.size(rfp_fa))), np.zeros(np.size(rfp_fa)),
                    np.array([[1]]), 0, b1_ax.reshape(np.size(b1_ax), 1))
Mxyfa = 2 * np.conj(a) * b

# shift pulse
tb=4
pbc=1.1
rfp_bs_shift, rfp_ss_shift, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_shift = rfp_bs_shift+rfp_ss_shift
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_shift.reshape((1, np.size(rfp_shift))), np.zeros(np.size(rfp_shift)),
                    np.array([[1]]), 0, b1_ax.reshape(np.size(b1_ax), 1))
Mxyshift = 2 * np.conj(a) * b

# tb pulse
tb=1.5
pbc=1.4
rfp_bs_tb, rfp_ss_tb, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_tb = rfp_bs_tb+rfp_ss_tb
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_tb.reshape((1, np.size(rfp_tb))), np.zeros(np.size(rfp_tb)),
                    np.array([[1]]), 0, b1_ax.reshape(np.size(b1_ax), 1))
Mxytb = 2 * np.conj(a) * b

# analyze the ratios of the signal in experimental and simulated profiles
indices_exp_pbc = np.where((b1_ax < pbc+pbw/4) & (b1_ax > pbc-pbw/4))
mean_exp = fa_d[indices_exp_pbc] / base_d[indices_exp_pbc]
pl.LinePlot(mean_exp)
mean_sim = Mxyfa[indices_exp_pbc] / Mxybase[indices_exp_pbc]
pl.LinePlot(mean_sim)
mean_fa_scale_exp = np.average(mean_exp)
mean_fa_scale_sim = np.average(mean_sim)




fig, axs = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,10))

experimental_ymax = 4E8
sim_ymax = 1.15 # gets the 90s roughly even
# shifting plots
axs[0].set_title('Simulated')
axs[1].set_title('Experimental')
axs[0].set_ylabel(r'$|M_{xy}|/M_0$')
axs[1].set_ylabel(r'signal (a.u.)')
axs[0].set_xlabel(r'$B_1^+ (mT)$')
axs[1].set_xlabel(r'$B_1^+ (mT)$')


# shifting
axs[0].plot(b1_ax.T,abs(Mxybase.T))
# axs[0].plot(b1_ax.T,abs(Mxyshift.T))
axs[1].plot(b1_ax.T,base_d.T)
axs[1].plot(b1_ax.T,shift_d.T)
axs[0].set_ylim([0,sim_ymax])
axs[1].set_ylim([0,experimental_ymax])
# axs[1].legend([r"PBC=1.4G",r"PBC=1.1G"],loc='center left',bbox_to_anchor=(1,0.5))


# # TB plots
# axs[2,0].plot(b1_ax.T,abs(Mxybase.T))
# axs[2,0].plot(b1_ax.T,abs(Mxytb.T),'r')
# axs[2,1].plot(b1_ax.T,base_d.T)
# axs[2,1].plot(b1_ax.T,tb_d.T,'r')
# axs[2,0].set_ylim([0,sim_ymax])
# axs[2,1].set_ylim([0,experimental_ymax])
# axs[2,1].legend([r'TB=4',r'TB=1.5'],loc='center left',bbox_to_anchor=(1,0.5))


# # FA plots
# axs[1,0].plot(b1_ax.T,abs(Mxybase.T))
# axs[1,0].plot(b1_ax.T,abs(Mxyfa.T),'g')
# axs[1,1].plot(b1_ax.T,base_d.T)
# axs[1,1].plot(b1_ax.T,fa_d.T,'g')
# axs[1,0].set_ylim([0,sim_ymax])
# axs[1,1].set_ylim([0,experimental_ymax])
# axs[1,1].legend([r'FA=90$^\circ$',r'FA=45$^\circ$'],loc='center left',bbox_to_anchor=(1,0.5))

plt.savefig('figures_out/experimental_ISMRM.png',dpi=300)

plt.show()