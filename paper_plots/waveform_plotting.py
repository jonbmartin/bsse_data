import numpy as np
import sigpy.mri.rf as rf
import sigpy.plot as pl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
dt = 1e-6

pbc = 1.4
pbw = 0.3

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=4, ndes=256, ptype='ex',
                                   flip=np.pi / 2, pbw=pbw, pbc=[pbc],
                                   d1e=0.01, d2e=0.01, rampfilt=True,
                                   bs_offset=7500)

# pl.LinePlot(np.real(rfp_ss)**2+np.imag(rfp_ss)**2)
T = np.size(rfp_ss)*dt
print(f'pulse_duration = {T * 1000} ms' )
t = np.linspace(- np.int(T / dt / 2), np.int(T / dt / 2), np.size(rfp_ss))
rfp_modulation=2087 # Hz
#pl.LinePlot(rfp_ss/np.exp(-1j * 2 * np.pi * rfp_modulation * t * dt))
am_bs = np.abs(rfp_bs)
am_bs = np.ones(np.shape(am_bs)) # TODO CHANGE
am_ss = rfp_ss/np.exp(-1j * 2 * np.pi * rfp_modulation * t * dt)
print(am_ss[:,501])
am_ss[:,500:np.size(am_ss)-500] = am_ss[:,500:np.size(am_ss)-500]


fm_bs = np.diff(np.unwrap(np.imag(np.log((rfp_bs+rfp_ss) / am_bs))))/(dt*2*np.pi)  # Hz
fm_ss = - np.ones(np.size(fm_bs)) * rfp_modulation  # Hz
fm_bs = 2*(fm_bs) - 7500 # TODO CHANGE

# pl.LinePlot(rfp_ss)
am_bs = am_bs[:, :-1]
am_ss = am_ss[:, :-1]

t = np.arange(0,dt*np.size(am_bs),dt)*1000  # ms
### PLOTTING######

color_am = '#26495c'
color_am = 'k'
color_fm = '#c66b3d'

# Plotting BS pulse (kHz FM)
fig, (ax1, ax3) = plt.subplots(1,2,figsize = [12,4])
ax1.plot(t, np.squeeze(am_bs), color_am)
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('AM (a.u.)', color=color_am)
ax1.set_title(r'Frequency Shift Inducing $b_{bs}(t)$')
ax1.set_title(r'BSSE initializer pulse')

ax1.tick_params(axis='y', labelcolor=color_am)
ax2 = ax1.twinx()
ax2.plot(t, np.squeeze(fm_bs)/1000, color_fm)
ax2.set_ylabel(r'$\Delta \omega$ (kHz)', color=color_fm)
ax2.tick_params(axis='y', labelcolor=color_fm)
ax2.set_ylim([0, np.max(abs(fm_bs))/1000])


# Plotting SS pulse
ax3.plot(t, np.squeeze(np.real(am_ss)), color_am)
ax3.set_xlabel('time (ms)')
ax3.set_ylabel('AM (a.u.)', color=color_am)
ax3.set_title(r'Frequency-Selective $b_{ex}(t)$')

ax3.tick_params(axis='y', labelcolor=color_am)
ax4 = ax3.twinx()
ax4.set_ylabel(r'$\Delta \omega$ (kHz)', color=color_fm)
ax4.tick_params(axis='y', labelcolor=color_fm)
ax4.plot(t, np.squeeze(fm_ss)/1000, color_fm)

ax4.set_ylim([0, -(np.max(abs(fm_bs)))/1000])

full_pulse = rfp_bs+rfp_ss
pwr_ss = (np.sum(abs(rfp_ss)))**2*dt
pwr_ss = integrate.trapz(abs(rfp_ss)**2, dx=dt)
pwr_bs = (np.sum(abs(rfp_bs)))**2*dt
pwr_bs = integrate.trapz(abs(rfp_bs)**2, dx=dt)
pwr_total= (np.sum(abs(rfp_bs+rfp_ss)))**2*dt
pwr_total = integrate.trapz(abs(rfp_ss+rfp_bs)**2, dx=dt)

print(f'power ratio = {(pwr_bs*((2022550+7500)/(2022550))**2+pwr_ss)/pwr_ss}')

b1 = np.arange(0, 3, 0.01)  # gauss, b1 range to sim over
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b
color_b1 = 'k'

# ax5.plot(b1, np.abs(Mxyfull.transpose()), color_b1)
# ax5.set_title('|Mxy|', color=color_b1)
# ax5.set_xlabel(r'$B_1$ (Gauss)', color=color_b1)
# ax5.tick_params(axis='y', labelcolor=color_b1)
# ax5.tick_params(axis='x', labelcolor=color_b1)

fig.tight_layout()
# plt.subplots_adjust(bottom=0.3, top=0.7,left=0.1, right=0.2, hspace=0)
plt.show()

plt.figure()
t = np.arange(0,dt*(np.size(am_bs)+1),dt)*1000  # ms
plt.plot(t, np.squeeze(abs(full_pulse).T),color=color_am)
plt.xlabel('time (ms)')
plt.ylabel('AM (a.u.)')
plt.title(r'Full BSSE waveform, $|b_{total}(t)|$')
plt.show()

### plotting the full pulse and its profile

color_rf = '#1e2761'
color_rf = 'k'
color_b1 = '#7a2048'
full_pulse = rfp_bs+rfp_ss
b1 = np.arange(0, 3, 0.01)  # gauss, b1 range to sim over
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

t = np.arange(0, dt*np.size(rfp_ss), dt)*1000  # ms

# Plotting summation pulse
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.plot(t, np.squeeze(np.abs(rfp_ss+rfp_bs)), color_rf)
ax1.set_xlabel('time (ms)', color=color_rf)
ax1.set_title('|RF| (a.u.)', color=color_rf)
ax1.tick_params(axis='y', labelcolor=color_rf)
ax1.tick_params(axis='x', labelcolor=color_rf)

ax2.plot(b1, np.abs(Mxyfull.transpose()), color_b1)
ax2.set_title('|Mxy|', color=color_b1)
ax2.set_xlabel(r'$B_1$ (Gauss)', color=color_b1)
ax2.tick_params(axis='y', labelcolor=color_b1)
ax2.tick_params(axis='x', labelcolor=color_b1)


fig.tight_layout()
plt.show()

# Plotting the summed excitation pulse

