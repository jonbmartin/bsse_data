import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import scipy.io as sio
dt = 2e-6

### PULSE DESIGNS
## PULSE 1
pbc = 3
pbw = 0.3
tb = 4
bsse_pulse1, rfp, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                     pbc=[pbc], d1e=0.01, d2e=0.01, rampfilt=True, bs_offset=16370)
bsse_pulse1 = bsse_pulse1 + rfp

print('pulse1 BSSE pulse length = {}s'.format(np.size(bsse_pulse1) * dt))

# design the RFSE pulse
[rfp_rfse_am1, rfp_rfse_fm1] = rf.dz_b1_rf(dt=dt, tb=tb, ptype='ex', flip=np.pi / 2, pbw=pbw, pbc=pbc,
                                           d1=0.01, d2=0.01, os=8, split_and_reflect=True)
print('pulse1 RFSE pulse length = {}s'.format(np.size(rfp_rfse_am1) * dt))

### PULSE 2
pbc = 3
pbw = 0.6
tb = 4
bsse_pulse2, rfp, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                     pbc=[pbc], d1e=0.01, d2e=0.01, rampfilt=True, bs_offset=9485)
bsse_pulse2 = bsse_pulse2 + rfp

print('pulse2 BSSE pulse length = {}s'.format(np.size(bsse_pulse2) * dt))

# design the RFSE pulse
[rfp_rfse_am2, rfp_rfse_fm2] = rf.dz_b1_rf(dt=dt, tb=tb, ptype='ex', flip=np.pi / 2, pbw=pbw, pbc=pbc,
                                           d1=0.01, d2=0.01, os=8, split_and_reflect=True)
print('pulse2 RFSE pulse length = {}s'.format(np.size(rfp_rfse_am2) * dt))

### PULSE 3
pbc = 3
pbw = 0.3
tb = 8
bsse_pulse3, rfp, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                     pbc=[pbc], d1e=0.01, d2e=0.01, rampfilt=True, bs_offset=19250)
bsse_pulse3 = bsse_pulse3 + rfp

print('pulse3 BSSE pulse length = {}s'.format(np.size(bsse_pulse3) * dt))

# design the RFSE pulse
[rfp_rfse_am3, rfp_rfse_fm3] = rf.dz_b1_rf(dt=dt, tb=tb, ptype='ex', flip=np.pi / 2, pbw=pbw, pbc=pbc,
                                           d1=0.01, d2=0.01, os=8, split_and_reflect=True)
print('pulse3 RFSE pulse length = {}s'.format(np.size(rfp_rfse_am3) * dt))

### PULSE 4
pbc = 1.5
pbw = 0.3
tb = 4
bsse_pulse4, rfp, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                     pbc=[pbc], d1e=0.01, d2e=0.01, rampfilt=True, bs_offset=8170)
bsse_pulse4 = bsse_pulse4 + rfp

print('pulse4 BSSE pulse length = {}s'.format(np.size(bsse_pulse4) * dt))

# design the RFSE pulse
[rfp_rfse_am4, rfp_rfse_fm4] = rf.dz_b1_rf(dt=dt, tb=tb, ptype='ex', flip=np.pi / 2, pbw=pbw, pbc=pbc,
                                           d1=0.01, d2=0.01, os=8, split_and_reflect=True)
print('pulse4 RFSE pulse length = {}s'.format(np.size(rfp_rfse_am4) * dt))

#SIMULATION
#general parameters
# do a simulation across b1 + b0 - first of the RFSE pulse
minb0, maxb0 = 5000, 10000
minb1, maxb1 = 0, 5
domdt = np.arange(minb0, maxb0, 10) * 2 * np.pi * dt
b1 = np.arange(minb1, maxb1, 0.05) # gauss, b1 range to sim over
# low-res to test with:
# domdt = np.arange(minb0, maxb0, 100) * 2 * np.pi * dt
# b1 = np.arange(minb1, maxb1, 0.1) # gauss, b1 range to sim over
domdtMat, b1Mat = np.meshgrid(domdt, b1)
newsim = True
if newsim:

    ##pulse 1
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_rfse_am1).reshape((1, np.size(rfp_rfse_am1))), 2 * np.pi * dt * np.reshape(rfp_rfse_fm1, (np.size(rfp_rfse_am1), 1)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_rfse1 = MxyMat.reshape((len(b1), len(domdt)))

    # finally, simulate the BSSE pulse
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (bsse_pulse1).reshape((1, np.size(bsse_pulse1))), np.zeros(np.size(bsse_pulse1)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_bs1 = MxyMat.reshape((len(b1), len(domdt)))

    ##pulse 2
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_rfse_am2).reshape((1, np.size(rfp_rfse_am2))), 2 * np.pi * dt * np.reshape(rfp_rfse_fm2, (np.size(rfp_rfse_am2), 1)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_rfse2 = MxyMat.reshape((len(b1), len(domdt)))

    # finally, simulate the BSSE pulse
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (bsse_pulse2).reshape((1, np.size(bsse_pulse2))), np.zeros(np.size(bsse_pulse2)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_bs2 = MxyMat.reshape((len(b1), len(domdt)))

    ##pulse 3
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_rfse_am3).reshape((1, np.size(rfp_rfse_am3))), 2 * np.pi * dt * np.reshape(rfp_rfse_fm3, (np.size(rfp_rfse_am3), 1)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_rfse3 = MxyMat.reshape((len(b1), len(domdt)))

    # finally, simulate the BSSE pulse
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (bsse_pulse3).reshape((1, np.size(bsse_pulse3))), np.zeros(np.size(bsse_pulse3)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_bs3 = MxyMat.reshape((len(b1), len(domdt)))

    ##pulse 4
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_rfse_am4).reshape((1, np.size(rfp_rfse_am4))), 2 * np.pi * dt * np.reshape(rfp_rfse_fm4, (np.size(rfp_rfse_am4), 1)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_rfse4 = MxyMat.reshape((len(b1), len(domdt)))

    # finally, simulate the BSSE pulse
    a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (bsse_pulse4).reshape((1, np.size(bsse_pulse4))), np.zeros(np.size(bsse_pulse4)),
                      np.array([[1]]), domdtMat.flatten(), b1Mat.reshape((len(b1Mat.flatten()), 1)))
    MxyMat = 2 * np.conj(a) * b
    MxyMat_bs4 = MxyMat.reshape((len(b1), len(domdt)))

    sio.savemat('offres_sim.mat',{'MxyMat_bs1':MxyMat_bs1,
                                  'MxyMat_bs2':MxyMat_bs2,
                                  'MxyMat_bs3':MxyMat_bs3,
                                  'MxyMat_bs4':MxyMat_bs4,
                                  'MxyMat_rfse1':MxyMat_rfse1,
                                  'MxyMat_rfse2':MxyMat_rfse2,
                                  'MxyMat_rfse3':MxyMat_rfse3,
                                  'MxyMat_rfse4':MxyMat_rfse4,
                                  })

if not newsim:
    data = sio.loadmat('offres_sim.mat')
    MxyMat_bs1 = data['MxyMat_bs1']
    MxyMat_bs2 = data['MxyMat_bs2']
    MxyMat_bs3 = data['MxyMat_bs3']
    MxyMat_bs4 = data['MxyMat_bs4']
    MxyMat_rfse1 = data['MxyMat_rfse1']
    MxyMat_rfse2 = data['MxyMat_rfse2']
    MxyMat_rfse3 = data['MxyMat_rfse3']
    MxyMat_rfse4 = data['MxyMat_rfse4']

# get both into kHz (just makes plot cleaner
# maxb0 /= 1000
# minb0 /= 1000

fig, axs = pyplot.subplots(nrows=4, ncols=2, sharex=True, sharey=True,figsize=(9.5,14))

aspect = 150
fig_bs1 = axs[0,0].imshow(abs(MxyMat_bs1), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)
axs[0,0].set_title('BSSE', size=14)
axs[0,0].set_ylabel(r'$B_1^+$ (mT)')
fig_rfse1 = axs[0,1].imshow(abs(MxyMat_rfse1), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)
axs[0,0].set_yticklabels(['0','0.1','0.3','0.5'])
axs[0,1].set_title('RFSE', size=14)

fig_bs2 = axs[1,0].imshow(abs(MxyMat_bs2), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)
axs[1,0].set_ylabel(r'$B_1^+$ (mT)')
axs[1,0].set_yticklabels(['0','0.1','0.3','0.5'])
fig_rfse2 = axs[1,1].imshow(abs(MxyMat_rfse2), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)

fig_bs3 = axs[2,0].imshow(abs(MxyMat_bs3), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)
axs[2,0].set_ylabel(r'$B_1^+$ (mT)')
axs[2,0].set_yticklabels(['0','0.1','0.3','0.5'])
fig_rfse3 = axs[2,1].imshow(abs(MxyMat_rfse3), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)

fig_bs4 = axs[3,0].imshow(abs(MxyMat_bs4), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)
axs[3,0].set_ylabel(r'$B_1^+$ (mT)')
axs[3,0].set_yticklabels(['0','0.1','0.3','0.5'])
fig_rfse4 = axs[3,1].imshow(abs(MxyMat_rfse4), origin='lower', extent=[minb0, maxb0, minb1, maxb1], aspect=aspect, vmin=0, vmax=1)

axs[3,0].set_xlabel(r'$\Delta f$ (Hz)')
axs[3,1].set_xlabel(r'$\Delta f$ (Hz)')
# space out so no overlap. Share axis so only need to change 1 for x and y
xticks = [0, 300, 600, 900]
yticks = [0, 1, 3, 5]
axs[3,0].set_xticks(xticks)
axs[1,0].set_yticks(yticks)


label_offset = -1.6
axs[0,0].text(label_offset, 0.5, "TB 4 \nPBC=0.3 mT\nPBW=0.03 mT\n6.27 ms",
              horizontalalignment='center', verticalalignment='center',
              transform=axs[0,1].transAxes)
axs[1,0].text(label_offset, 0.5, r"PBW$\to$0.06 mT"+"\n3.14 ms",
              horizontalalignment='center', verticalalignment='center',
              transform=axs[1,1].transAxes)
axs[2,0].text(label_offset, 0.5, r"TB$\to$8"+"\n12.51 ms",
              horizontalalignment='center', verticalalignment='center',
              transform=axs[2,1].transAxes)
axs[3,0].text(label_offset, 0.5, r"PBC$\to$0.15 mT"+"\n6.27 ms",
              horizontalalignment='center', verticalalignment='center',
              transform=axs[3,1].transAxes)
fig.tight_layout(pad=4)


pyplot.subplots_adjust(wspace=0.075, hspace=0.075)
# pyplot.pcolor(abs(MxyMat_rfse), cmap='viridis', vmin=0, vmax=1)
# ax2.colorbar(location='right')# axs[1].colorbar(location='right')
cbar = fig.colorbar(fig_bs4, ax=axs.ravel().tolist(), shrink=0.7)
cbar.set_label(r'$|M_{xy}|/M_0$',size=14)
pyplot.setp(axs[0,1].get_yticklabels(), visible=False)
# axs[1, 0].imshow(abs(MxyMat_bs_lowb1), origin='lower')
# axs[1, 1].imshow(abs(MxyMat_rfse_lowb1), origin='lower')
# fig.colorbar(axs[1].pcolormesh(abs(MxyMat_bs)))



pyplot.savefig('figures_out/offres.png',dpi=300)
pyplot.show()
