import numpy as np
import matplotlib.pyplot as plt

bs_offset = 20000
dt = 4e-6
T = 0.007
dw0 = 2 * np.pi * bs_offset  # amplitude of sweep
nsw = np.round(500e-6 / dt)  # number of time points in sweeps

beta=4
kappa=np.arctan(4)

# build the AHP pulse
t = range(np.int(nsw)) / nsw
a = np.expand_dims(np.tanh(beta * t), 0)
om = np.expand_dims(dw0 * np.tan(kappa * (t - 1)) / np.tan(kappa), 0)

n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

# build the complete ,BS pulse
bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)

bs_fm = np.concatenate([-om, np.zeros((1, n)),
                        -np.fliplr(om)], axis=1) / 2 / np.pi + bs_offset

t = np.arange(0,dt*np.size(bs_am),dt)
fig, ax1 = plt.subplots()
ax1.plot(t, np.squeeze(bs_am),'k')

ax2 = ax1.twinx()
ax2.plot(t, np.squeeze(bs_fm),'gray')

ax2.set_ylim([0, 40000])

plt.show()