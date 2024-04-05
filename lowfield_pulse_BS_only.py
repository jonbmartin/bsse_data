import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 1e-6
b1 = np.arange(0, 1.5, 0.01)  # gauss, b1 range to sim over

pbc = 1.3
pbw = 0.3
bs_offset = 7500
tb = 8
# actual design that will get exported to TNMR
rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=0.01, d2e=0.01,
                                   rampfilt=True, bs_offset=bs_offset)

# sweep parameters for BS pulse
nsw = np.round(500e-6 / dt)  # number of time points in sweeps
beta = 0.5  # AM waveform parameter, you want this to reach ~1 @ end of pulse
dw0 = 2 * np.pi * bs_offset  # amplitude of sweep# build the AHP pulse
kappa = np.arctan(4)  # FM waveform parameter

# calculate bandwidth and pulse duration using lowest PBC of bands. Lower
# PBC's require a longer pulse, so lowest constrains our pulse length
upper_b1 = pbc + pbw / 2
lower_b1 = pbc - pbw / 2

# using Ramsey's BS shift equation pre- w_rf >> gam*b1 approximation
B = bs_offset * (
            (1 + (4258 * upper_b1) ** 2 / bs_offset ** 2) ** (1 / 2) - 1) - \
    bs_offset * ((1 + (4258 * lower_b1) ** 2 / bs_offset ** 2) ** (1 / 2) - 1)
T = (tb / B)  # seconds, the entire pulse duration

t = range(np.int(nsw)) / nsw
sigma = beta / 6
a = 1 / (1 + np.exp((t - beta) / sigma))
a = np.fliplr(np.expand_dims(a, 0))
om = np.expand_dims(dw0 * np.tan(kappa * (t - 1)) / np.tan(kappa), 0)

n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

# build the complete BS pulse
bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)
n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

# build the complete BS pulse
bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)

bs_fm = np.concatenate([-om, np.zeros((1, n)),
                        -np.fliplr(om)], axis=1) / 2 / np.pi + bs_offset

pl.LinePlot(bs_fm)
print(np.mean(bs_fm))

fm_dic = {"bs_fm":bs_fm}
sio.savemat('BS_only_10khz_FM.mat', fm_dic)

bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))

# calculate the kbs of the pulse
kbs = rf.calc_kbs(bs_am, bs_fm, dt*np.size(bs_fm))
print('Kbs = {}'.format(kbs))
print('Pulse duration = {} s'.format(np.size(bsrf)*dt))


quantized_am = np.abs(bs_am) / np.max(np.abs(bs_am))
scalefact = 1  # 39 has worked well in past, but to be consistent with b1 mapping
quantized_am *= scalefact
print(np.mean(quantized_am))

pl.LinePlot(quantized_am, title='AM out')

# quantized_bs = abs(rfp_bs)/np.max(abs(full_pulse))
# quantized_ss = abs(rfp_ss)/np.max(abs(full_pulse))
phaseout = np.angle(bsrf) * (180/np.pi) + 180 # 0 to 360
# pl.LinePlot(phaseout, title='Phase out')
# pl.LinePlot(bs_fm)

pulse_dic = {"bs_only_am":quantized_am, "bs_only_phase":phaseout,  "kbs":kbs}
sio.savemat('BS_NEW_only_10khz_scalefact1.mat', pulse_dic)


full_pulse = rfp_bs + rfp_ss
