import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 1e-6
b1 = np.arange(0,8, 0.01)  # gauss, b1 range to sim over

pbc = 6
pbw = 2

am, fm = rf.dz_b1_rf(dt=dt, tb=4, ptype='st', flip=np.pi / 6, pbw=pbw,
             pbc=pbc, d1=0.01, d2=0.01, os=8, split_and_reflect=True)

bsrf = am * np.exp(1j * dt * 2 * np.pi * np.cumsum(fm))

print('rfse dur = {} s'.format(np.size(am)*dt*1000 ))

a, b = rf.abrm_hp(2*np.pi*4258*dt*bsrf.reshape((1, np.size(bsrf))), np.zeros(np.size(bsrf)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))
Mxyfull = 2 * np.conj(a) * b

pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.show()


####### COMPARING SCALING METHODS #######
# quantized_am1 = np.abs(rfp_bs) / np.max(np.abs(rfp_bs))
# quantized_am1 = quantized_am1 / np.mean(quantized_am1)
# scalefact = 17
# quantized_am1 *= scalefact
# pl.LinePlot(quantized_am1, title='mapping pulse fully scaled')
# print(np.max(abs(quantized_am1)))
########################33

quantized_am = np.abs(am) / np.max(np.abs(am))
quantized_am = quantized_am / np.mean(quantized_am)
scalefact = 17
quantized_am *= scalefact
pl.LinePlot(quantized_am, title='bsse pulse fully scaled')
# pl.LinePlot(quantized_am)
# quantized_bs = abs(rfp_bs)/np.max(abs(full_pulse))
# quantized_ss = abs(rfp_ss)/np.max(abs(full_pulse))
phaseout = np.angle(full_pulse) * (180/np.pi) + 180 # 0 to 360


pulse_dic = {"bsse_am":quantized_am.T, "bsse_phase":phaseout.T, "raw_bs":rfp_bs, "raw_ss":rfp_ss}
sio.savemat('base_bsse_pulse_0d9G_nomean_10khz.mat', pulse_dic)