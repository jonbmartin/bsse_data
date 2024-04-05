import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio
import mat73

dt = 1e-6
db1 = 0.01
pbc = 2
b1min = 0
b1max = pbc+1
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

pbw = 0.3 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 2
bs_offset = 7500

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)
data_input = sio.loadmat('bsse_singleband_library/bsse_pulse_flatex_50khzoff.mat')

data_output = mat73.loadmat('bsse_singleband_library/flat_optimized_pulse_50khz_offset.mat')
rfp_angle_unrefined = np.angle(data_input['rfp_bs'] + data_input['rfp_ss'])
rfp_angle = data_output['rfp_angle']
fm_bs_refined = np.diff(np.unwrap(rfp_angle))/(dt*2*np.pi)
fm_bs_unrefined = np.diff(np.unwrap(rfp_angle_unrefined))/(dt*2*np.pi)

pyplot.plot(fm_bs_unrefined.T)
pyplot.plot(fm_bs_refined.T)
pyplot.legend(['unrefined', 'refined'])
pyplot.xlabel('samples')
pyplot.ylabel('FM (Hz)')

pyplot.show()
# pl.LinePlot(rfp_bs)
# pl.LinePlot(rfp_ss)
full_pulse = rfp_bs + rfp_ss
full_pulse = np.ones(np.size(rfp_angle)) * np.exp(1j*rfp_angle)
# full_pulse = full_pulse / abs(full_pulse)
print(f'pulse duration = {np.size(rfp_ss)*dt*1000} ms')
pl.LinePlot(full_pulse)

print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

w = np.zeros(np.size(b1))

pbwtop_index = int((pbc+pbw/2-b1min)/db1)+1
pbwbot_index = int((pbc-pbw/2-b1min)/db1)

w[pbwbot_index:pbwtop_index] = 1
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.plot(b1, w)
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.ylim([0,1.05])
pyplot.xlim([b1min,b1max])
pyplot.show()
# pl.LinePlot(Mxyfull)
rfp_bs_am = abs(rfp_bs)
rfp_bs_p = np.angle(rfp_bs)
rfp_ss_am = abs(rfp_ss)
rfp_ss_p = np.angle(rfp_ss)

# off-resonance evaluation
minb0, maxb0 = 0, 1000
minb1, maxb1 = 0, 4
domdt = np.arange(minb0, maxb0, 10) * 2 * np.pi * dt
b1 = np.arange(minb1, maxb1, 0.05) # gauss, b1 range to sim over

domdtMat, b1Mat = np.meshgrid(domdt, b1)

a, b = rf.abrm_hp(
    2 * np.pi * 4258 * dt * (full_pulse).reshape((1, np.size(full_pulse))),
    np.zeros(np.size(full_pulse)),
    np.array([[1]]), domdtMat.flatten(),
    b1Mat.reshape((len(b1Mat.flatten()), 1)))
MxyMat = 2 * np.conj(a) * b
MxyMat_bs1 = MxyMat.reshape((len(b1), len(domdt)))

# pyplot.imshow(abs(MxyMat_bs1), extent=[minb0, maxb0, minb1, maxb1])
aspect = 150
pyplot.imshow(abs(MxyMat_bs1), extent=[maxb0, minb0, minb1, maxb1], aspect=aspect)

# pyplot.colorbar()
pyplot.title('|Mxy| across off-resonance, |B1+|')
pyplot.xlabel('off-resonance (Hz)')
pyplot.ylabel('B1 (G)')


pyplot.show()