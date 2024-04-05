import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot

def predict_mp(om1, om2, N):
    '''

    :param om1: khz
    :param om2: khz
    :param N: multiphoton resonance number 
    :return: Nth resonance, in Gauss
    '''
    av = 0.5 * (om1 + om2)
    res_cond_num = 1 / N
    pbc_khz = N * abs(om1-av)-av
    pbc_hz = pbc_khz * 1000
    b1_mp = (om1*1000/4258)*np.sqrt((1+(pbc_hz)/(om1*1000))**2 - 1)
    return b1_mp

dt = 2e-6  # s
d1e = 0.05
d2e = 0.05
pbc = 1.4
pbw = 0.3
offset_bs = 7500
offset_shift = 10000
tb = 4
# CORRESPONDING SS MODULATION FOR A PBC=2 IS 3.135 kHz
# CORRESPONDING SS MODULATION FOR A PBC=1.4 IS 2.081 kHz
N = np.array([1, 3, 5, 7, 9, 11])
b1_mp = predict_mp(offset_shift/1000,-1.655,N)
print(f'Red MP resonances are located at {b1_mp}')
b1_mp = predict_mp(offset_bs/1000,-2.094,N)
print(f'Black MP resonances are located at {b1_mp}')

## DESIGN BASE PULSES
bs_pulse_pi2, ss_pulse_pi2, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex', flip=1 * np.pi / 2,
                                               pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                               rampfilt=True, bs_offset=offset_bs)

bs_pulse_inv, ss_pulse_inv, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='inv', flip=1 * np.pi,
                                               pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                               rampfilt=False, bs_offset=offset_bs)

bs_pulse_largeFA, ss_pulse_largeFA, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='st', flip=8*np.pi/2,
                                                       pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                                       rampfilt=True, bs_offset=offset_bs)
## DESIGN SHIFTED PULSES
bs_pulse_pi2_s, ss_pulse_pi2_s, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex', flip=1 * np.pi / 2,
                                               pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                               rampfilt=True, bs_offset=offset_shift)

bs_pulse_inv_s, ss_pulse_inv_s, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='inv', flip=1 * np.pi,
                                               pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                               rampfilt=False, bs_offset=offset_shift)

bs_pulse_largeFA_s, ss_pulse_largeFA_s, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='st', flip=8*np.pi/2,
                                                       pbw=pbw, pbc=[pbc], d1e=d1e, d2e=d2e,
                                                       rampfilt=True, bs_offset=offset_shift)

t = np.expand_dims(np.arange(0, np.size(bs_pulse_largeFA)*dt, dt) * 1000, 1)  # ms
t_s = np.expand_dims(np.arange(0, np.size(bs_pulse_largeFA_s)*dt, dt) * 1000, 1)  # ms
b1 = np.arange(0, 15, 0.01)  # gauss, b1 range to sim over
gam = 4258  # Hz/G

# SIM THE BASELINE MAGNETIZATION
a_90, b_90 = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_pi2 + ss_pulse_pi2), np.zeros(np.size(bs_pulse_pi2)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90 = np.squeeze(2 * np.conj(a_90) * b_90)

a_inv, b_inv = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_inv + ss_pulse_inv), np.zeros(np.size(bs_pulse_inv)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzInv = 1-2*np.abs(b_inv)**2

a_large, b_large = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_largeFA + ss_pulse_largeFA), np.zeros(np.size(bs_pulse_pi2)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy4pi = np.squeeze(2 * np.conj(a_large) * b_large)

# SIM THE SHIFTED
a_90, b_90 = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_pi2_s + ss_pulse_pi2_s), np.zeros(np.size(bs_pulse_pi2_s)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy90_s = np.squeeze(2 * np.conj(a_90) * b_90)

a_inv, b_inv = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_inv_s + ss_pulse_inv_s), np.zeros(np.size(bs_pulse_inv_s)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
MzInv_s = 1-2*np.abs(b_inv)**2

a_large, b_large = rf.abrm_hp(2 * np.pi * 4258 * dt * (bs_pulse_largeFA_s + ss_pulse_largeFA_s), np.zeros(np.size(bs_pulse_pi2_s)),
                        np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxy4pi_s = np.squeeze(2 * np.conj(a_large) * b_large)

fig, axs = pyplot.subplots(3, 2,figsize=(10,7))
shift_color = 'r'
base_color = 'k'
# axs[0, 0].plot(t_s, abs(bs_pulse_pi2_s + ss_pulse_pi2_s).T, shift_color)
axs[0, 0].plot(t, abs(bs_pulse_pi2 + ss_pulse_pi2).T, base_color)
axs[0, 0].set_title(r'$|b_{total}(t)|$, FA = 90$^\circ$')
axs[0, 0].set_xlabel(r't (ms)')
axs[0, 0].set_ylabel(r'(a.u.)')

# axs[0, 1].plot(b1/10, abs(Mxy90_s), shift_color)
axs[0, 1].plot(b1/10, abs(Mxy90), base_color)

axs[0, 1].set_ylabel(r'$|M_{xy}|/M_0$')
axs[0, 1].set_xlabel(r'$B_1^+$ (mT)')
axs[0, 1].legend([ r'$\omega_{off}$ = 7.5 kHz'])

# axs[1, 0].plot(t_s, abs(bs_pulse_inv_s + ss_pulse_inv_s).T, shift_color)
axs[1, 0].plot(t, abs(bs_pulse_inv + ss_pulse_inv).T, base_color)
axs[1, 0].set_title(r'$|b_{total}(t)|$, FA = 180$^\circ$')
axs[1, 0].set_xlabel(r't (ms)')
axs[1, 0].set_ylabel(r'(a.u.)')

# axs[1, 1].plot(b1/10, np.squeeze(np.real(MzInv_s)), shift_color)
axs[1, 1].plot(b1/10, np.squeeze(np.real(MzInv)), base_color)
axs[1, 1].set_ylim([-1.05, 1.05])
axs[1, 1].set_ylabel(r'$M_{z}/M_0$')
axs[1, 1].set_xlabel(r'$B_1^+$ (mT)')

# axs[2, 0].plot(t_s, abs(bs_pulse_largeFA_s + ss_pulse_largeFA_s).T, shift_color)
axs[2, 0].plot(t, abs(bs_pulse_largeFA + ss_pulse_largeFA).T, base_color)
axs[2, 0].set_title(r'$|b_{total}(t)|$, FA = 720$^\circ$')
axs[2, 0].set_xlabel(r't (ms)')
axs[2, 0].set_ylabel(r'(a.u.)')

# axs[2, 1].plot(b1/10, abs(Mxy4pi_s), shift_color)
axs[2, 1].plot(b1/10, abs(Mxy4pi), base_color)
axs[2, 1].set_ylabel(r'$|M_{xy}|/M_0$')
axs[2, 1].set_xlabel(r'$B_1^+$ (mT)')


pyplot.tight_layout()
pyplot.subplots_adjust(
                    wspace=0.2,
                    hspace=0.5)

pyplot.savefig('figures_out/multiphoton.png',dpi=300)
pyplot.show()

