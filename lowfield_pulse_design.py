import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio
import mat73

dt = 2e-6
db1 = 0.01
pbc = 1.0
b1min = 0
b1max = pbc+1
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

pbw = 0.25 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 2
bs_offset = 10000


rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)


full_pulse = rfp_bs + rfp_ss
print(f'pulse duration = {np.size(rfp_ss)*dt*1000} ms')
pl.LinePlot(full_pulse)
# comparison_pulse = sio.loadmat('bs_neg_ex_pos.mat')['b1']z
# pl.LinePlot(full_pulse-comparison_pulse)
# T = np.size(full_pulse)*dt
#
# full_pulse = rf.bssel_ex_slr(T, dt=1e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/4,
#                  pbw=0.2, pbc=0.4, d1e=0.01, d2e=0.01, rampfilt=True,
#                  bs_offset=50000)
# pl.LinePlot(full_pulse)


print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

# create a ROI for RMSE & BSSE - should exclude transition regions and MP
# w = np.ones(np.size(b1))
w = np.zeros(np.size(b1))
# exclude the transition region
# ftw_g = rf.dinf(d1, d2) / tb * pbw
# w[int((pbc - pbw / 2) / db1 - ftw_g / db1 / 2):int(
#     (pbc - pbw / 2) / db1 + ftw_g / db1 / 2)] = 0
# w[int((pbc + pbw / 2) / db1 - ftw_g / db1 / 2):int(
#     (pbc + pbw / 2) / db1 + ftw_g / db1 / 2)] = 0
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

####### COMPARING SCALING METHODS #######
# quantized_am1 = np.abs(rfp_bs) / np.max(np.abs(rfp_bs))
# quantized_am1 = quantized_am1 / np.mean(quantized_am1)
# scalefact = 17
# quantized_am1 *= scalefact
# pl.LinePlot(quantized_am1, title='mapping pulse fully scaled')
# print(np.max(abs(quantized_am1)))
########################33
scalefact = 1  # 39 has worked well in past, but to be consistent with b1 mapping
pulse_scaled_out = rfp_bs*scalefact+rfp_ss*scalefact
# pl.LinePlot(pulse_scaled_out)
# quantized_am = np.abs(full_pulse) / np.max(np.abs(full_pulse))
# quantized_am *= scalefact
# print(np.mean(quantized_am))

# pl.LinePlot(quantized_am/np.max(quantized_am), title='bsse pulse fully scaled')

phaseout = np.angle(full_pulse) * (180/np.pi) + 180 # 0 to 360f


# pulse_dic = {"bsse_cplx":rfp_bs+rfp_ss,"bsse_am":abs(pulse_scaled_out.T), "bsse_phase":phaseout.T, "raw_bs":rfp_bs, "raw_ss":rfp_ss}
# sio.savemat('bsse_cplx_pulse.mat', pulse_dic)
pulse_dic = {"bsse_cplx":rfp_bs+rfp_ss, "raw_bs":rfp_bs, "raw_ss":rfp_ss,"dt":dt}
sio.savemat('bsse_cplx_pulse_4us.mat', pulse_dic)

jl_pulse_dic = {"rfp_bs": rfp_bs, "rfp_ss":rfp_ss, "dt":dt}
sio.savemat('bsse_flatinit_1d0pbc_0d25pbw_90.mat.mat',jl_pulse_dic)