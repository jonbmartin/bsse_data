import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm
import sigpy.mri.rf as rf


def scalefact(rfp_modulation, bs_offset):
    return (0.3323*np.exp(-0.9655*(rfp_modulation/bs_offset))
                         + 0.6821*np.exp(-0.02331*(rfp_modulation/bs_offset)))

def bs_freq_correction(pbc, bs_offset):
    bso = bs_offset * 1000 # convert to Hz from kHz
    rfp_correction_42 = 8.177 - 5.43 * pbc - 6.465E-4 * bso - 0.1134 * pbc ** 2 \
                        + 1.814E-4 * pbc * bso + 8.474E-9 * bso ** 2 + 0.02363 * pbc ** 3 \
                        - 2.177E-5 * pbc ** 2 * bso - 2.497E-9 * pbc * bso ** 2
    rfp_correction_42 = -4.632 -2.78*pbc + 4.043E-4*bso-0.2765*pbc**2 \
                        -6.776E-5*pbc*bso -4.25E-9*bso**2+0.0573*pbc**3 \
                        - 2.427E-5*pbc**2*bso +1.602E-9*pbc*bso**2 \
                        - 1.67E-3*pbc**4 + 7.705E-7 * pbc**3*bso \
                        - 1.994E-10 * pbc**2*bso**2
    return rfp_correction_42


data = sio.loadmat('empirical_error_data_UPDATED_SWEEP_101021.mat')
b1 = data['b1']
b1scale = data['b1scale']
bs = data['bs']
ss = data['ss']
ssbs = data['ssbs']
ssshift = data['ssshift']

bs_off_range = 10000
ss_off_range = np.linspace(0,8*bs_off_range,1000)



# pulses to simulate:
dt = 1e-6
db1 = 0.01
b1_sim = np.arange(0,3, db1)  # gauss, b1 range to sim over

pbc = 1.4  # b1 (Gauss)
pbw = 0.3 # b1 (Gauss)
bs_offset = 7500
ss_offset = 2080

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=0.01, d2e=0.01,
                                   rampfilt=True, bs_offset=7500,fa_correct=True, ss_correct=True)
full_corr = rfp_bs + rfp_ss
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_corr.reshape((1, np.size(full_corr))), np.zeros(np.size(full_corr)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxyfull_corr = 2 * np.conj(a) * b

rfp_bs_nofa, rfp_ss_nofa, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=0.01, d2e=0.01,
                                   rampfilt=True, bs_offset=7500,fa_correct=False, ss_correct=True)

full_nofa = rfp_bs_nofa + rfp_ss_nofa
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_nofa.reshape((1, np.size(full_nofa))), np.zeros(np.size(full_nofa)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxyfull_nofa = 2 * np.conj(a) * b

rfp_bs_noss, rfp_ss_noss, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=0.01, d2e=0.01,
                                   rampfilt=True, bs_offset=7500,fa_correct=True, ss_correct=False)

full_noss = rfp_bs_noss + rfp_ss_noss
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_noss.reshape((1, np.size(full_noss))), np.zeros(np.size(full_noss)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxyfull_noss = 2 * np.conj(a) * b

# pl.LinePlot(rfp_bs)
# pl.LinePlot(rfp_ss)
full_pulse = rfp_bs + rfp_ss
# #===============
# #  First subplot
# #===============
# ax = fig.add_subplot(2,2,1)
# ax.scatter(abs(ss/bs), b1scale/b1)
# ax.plot(ss_off_range/bs_off_range,scalefact(ss_off_range,bs_off_range),zorder=-1)
# ax.scatter(ss_offset/bs_offset,scalefact(ss_offset,bs_offset), c='r', zorder=2) # the experimental point
# ax.set_xlabel(r'$\omega_{cent}/\omega_{off}$')
# ax.set_ylabel(r'$B_1^+$ attenuation factor ($\Gamma$)')
# #===============
# #  2nd subplot
# #===============
# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.view_init(20, 65)
#
# b1_range = np.linspace(0,20,100)  # G
# bs_range = np.linspace(0,100,100)  # BS, khz
# b1_range, bs_range = np.meshgrid(b1_range,(bs_range))
# ax.plot_surface(b1_range,bs_range,bs_freq_correction(b1_range,bs_range)/1000,cmap=cm.viridis,zorder=-1)
# ax.scatter(b1,bs/1000,ssshift/1000,zorder=1)
# ax.scatter(pbc,bs_offset/1000,-bs_freq_correction(pbc,bs_offset/1000)/1000+0.025, c='r', zorder=2) # the experimental point. An artificial boost is given to make it visible
#
# ax.set_xlabel(r'$PBC$ (G)')
# ax.set_ylabel(r'$\omega_{off}$ (kHz)')
# ax.set_zlabel(r'$\Delta\omega_{err}$ (kHz)')
#
# #===============
# #  Magnetization no FA correction
# #===============
# ax = fig.add_subplot(2, 2, 3)
# x = b1_sim
# y1 = np.squeeze(abs(Mxyfull_nofa))
# y2 = np.squeeze(abs(Mxyfull_corr))
# ax.plot(x, y1)
# ax.plot(x, y2)
# ax.set_xlabel('$B_1^+$ (G)')
# ax.set_ylabel('$|M_{xy}|$')
#
# sidelength = 0.38
# miniplot_x = 0.675
# miniplot_y = 0.6
# axins = ax.inset_axes([miniplot_x,miniplot_y, sidelength, sidelength])
# xmin, xmax = 128, 151
#
#
# axins.plot(x[xmin: xmax],y1[xmin:xmax])
# axins.plot(x[xmin:xmax],y2[xmin:xmax])
# ax.indicate_inset_zoom(axins,linewidth=3)
# axins.set_xticklabels('')
# # axins.set_yticklabels('')
# ax.legend(['no FA \ncorrection', 'corrected'], loc='upper left')
#
#
# #===============
# #  Magnetization no ss correction
# #===============
# ax = fig.add_subplot(2, 2, 4)
# ax.plot(b1_sim,np.squeeze(abs(Mxyfull_noss)))
# ax.plot(b1_sim,np.squeeze(abs(Mxyfull_corr)))
# ax.set_xlabel('$B_1^+$ (G)')
# ax.set_ylabel('$|M_{xy}|$')
#
# x = b1_sim
# y1 = np.squeeze(abs(Mxyfull_noss))
# y2 = np.squeeze(abs(Mxyfull_corr))
# axins = ax.inset_axes([miniplot_x,miniplot_y, sidelength, sidelength])
# xmin, xmax = 150, 161
# xminb1 = xmin*db1
# xmaxb1 = xmax*db1
# axins.plot(x[xmin: xmax],y1[xmin:xmax])
# axins.plot(x[xmin:xmax],y2[xmin:xmax])
# ax.indicate_inset_zoom(axins,linewidth=3)
# # axins.set_xticklabels('')
# axins.set_yticklabels('')
#
# ax.legend(['no shift \ncorrection', 'corrected'], loc='upper left')
#
# plt.show()



###### SECOND VERSION - no omega_cent correction #######
fig = plt.figure(figsize=[10, 5])

#===============
#  First subplot
#===============
ax = fig.add_subplot(1,2,1)
ax.scatter(abs(ss/bs), b1scale/b1)
ax.plot(ss_off_range/bs_off_range,scalefact(ss_off_range,bs_off_range),zorder=-1)
ax.scatter(ss_offset/bs_offset,scalefact(ss_offset,bs_offset), c='r', zorder=2) # the experimental point
ax.set_xlabel(r'$\omega_{cent}/\omega_{off}$')
ax.set_ylabel(r'$B_1^+$ attenuation factor ($\Gamma$)')


#===============
#  Magnetization no FA correction
#===============
ax = fig.add_subplot(1,2, 2)
x = b1_sim/10
y1 = np.squeeze(abs(Mxyfull_nofa))
y2 = np.squeeze(abs(Mxyfull_corr))
ax.plot(x, y1)
ax.plot(x, y2)
ax.set_xlabel('$B_1^+$ (mT)')
ax.set_ylabel('$|M_{xy}|/M_0$')
ax.set_ylim([0, 1])
sidelength = 0.38
miniplot_x = 0.675
miniplot_y = 0.5
axins = ax.inset_axes([miniplot_x,miniplot_y, sidelength, sidelength])
xmin, xmax = 129, 152


axins.plot(x[xmin: xmax],y1[xmin:xmax])
axins.plot(x[xmin:xmax],y2[xmin:xmax])
ax.indicate_inset_zoom(axins,linewidth=3)
axins.set_xticklabels('')
# axins.set_yticklabels('')
ax.legend(['no FA \ncorrection', 'corrected'], loc='upper left')


plt.show()