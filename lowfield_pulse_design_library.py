import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio
import os

dt = 1e-6
b1 = np.arange(0,4, 0.01)  # gauss, b1 range to sim over
b1_centers = np.linspace(1,3,16)

pbw = 1/8 # b1 (Gauss)
dir = 'output_dir'
os.mkdir(dir)


pulse_durs = np.zeros(np.size(b1_centers))
for ii in range(np.size(b1_centers)):
    pbc = b1_centers[ii]

    rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                       pbc=[pbc], d1e=0.01, d2e=0.01,
                                       rampfilt=False, bs_offset=7500)
    pulse_durs[ii] = np.size(rfp_bs)*dt  # s
    full_pulse = rfp_bs + rfp_ss
    # pl.LinePlot(full_pulse)

    print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

    a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                        np.array([[1]]), 0, b1.reshape(np.size(b1), 1))

    Mxyfull = 2 * np.conj(a) * b
    # pyplot.figure()
    # pyplot.plot(b1, np.abs(Mxyfull.transpose()))
    # pyplot.ylabel('|Mxy|')
    # pyplot.xlabel('Gauss')
    # pyplot.ylim([0,1])
    # pyplot.show()

    rfp_bs_am = abs(rfp_bs)
    rfp_bs_p = np.angle(rfp_bs)
    rfp_ss_am = abs(rfp_ss)
    rfp_ss_p = np.angle(rfp_ss)

    scalefact = 1
    pulse_scaled_out = rfp_bs*scalefact+rfp_ss*scalefact

    phaseout = np.angle(full_pulse) * (180/np.pi) + 180 # 0 to 360


    pulse_dic = {"bsse_am":abs(pulse_scaled_out.T), "bsse_phase":phaseout.T,
                 "raw_bs":rfp_bs, "raw_ss":rfp_ss,"b1sim":b1,"mxy":Mxyfull,
                 "pbc":pbc,"pbw":pbw, "dur":pulse_durs[ii],"dt":dt}
    sio.savemat(dir+'/pulse_band_{}.mat'.format(ii), pulse_dic)

