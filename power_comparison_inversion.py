import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl

dt = 2e-6
pbc_min = 4
pbc_max = 1000
pbw_min = 0.2
pbw_max = 1.2
pbc = np.arange(pbc_min, pbc_max, 0.5)  # gauss, b1 range to sim over
pbw_list = np.arange(pbw_min, pbw_max, 0.1)
pwr_ratio_list = []
mse_ratio_list = []
t_bs = []
t_rfse = []

tb_start = 8
tb_interval = 0.1
err_thresh = 0.12
bs_offset = 15000

# PART 1 - sweep across PBC, hold PBW
pbw = 0.25
bsse_durs = []
rfse_durs = []
for pb in list(pbc):
    tb_bsse, tb_rfse = tb_start, tb_start
    mse_bs = np.inf
    print('##########################################')
    print('PBC = {} G, PBW = {} G'.format(pb, pbw))
    print('##########################################')
    print('BSSE pulse:')
    while mse_bs > err_thresh:

        bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb_bsse, ndes=128, ptype='se', flip=np.pi, pbw=pbw,
                                pbc=[pb], d1e=0.01, d2e=0.01, rampfilt=False, bs_offset=bs_offset)
        print('BSSE length = {} ms'.format(dt*np.size(bsrf)*1000))
        rfp_bs = bsrf + rfp_ex
        b1_sim = np.linspace(pb-2*pbw, pb+2*pbw, 100)
        db1 = (pb+2*pbw - (pb-2*pbw))/100
        ideal_prof = np.zeros(np.shape(b1_sim))
        ideal_prof[38:62] = 1
        # SIMULATE REFOCUSING EFFICIENCY
        a180, b180 = rf.abrm_hp(
            2 * np.pi * 4258 * dt * (bsrf + rfp_ex),
            np.zeros(np.size(bsrf.T)),
            np.array([[1]]), 0, b1_sim.reshape(len(b1_sim), 1))
        pyplot.figure()
        pyplot.title('BSSE Beta Squared')
        pyplot.plot(b1_sim, np.real(b180.T ** 2), label='BSSE Re(beta**2)')
        pyplot.plot(b1_sim, np.imag(b180.T ** 2), label='BSSE Im(beta**2)')
        pyplot.legend(loc='lower right')
        pyplot.xlabel('Gauss')
        pyplot.show()
    rfse_durs.append(dt*np.size(bsrf))  # seconds
    # pl.LinePlot(Mxybs,title='Final BSSE Mxy')

    print('RFSE pulse:')
    mse_rfse = np.inf
    while mse_rfse > err_thresh:

        [rfp_rfse_am, rfp_rfse_fm] = rf.dz_b1_rf(dt=dt, tb=tb_rfse, ptype='ex', flip=np.pi/2, pbw=pbw, pbc=pb,
                                               d1=0.01, d2=0.01, os=8, split_and_reflect=True)
        print('RFSE length = {} ms'.format(dt*np.size(rfp_rfse_am)*1000))
        rfse_rf = rfp_rfse_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(rfp_rfse_fm))

        [a, b] = rf.sim.abrm_nd(2 * np.pi * dt * rfp_rfse_fm, np.reshape(b1_sim, (np.size(b1_sim),1)),
                                2 * np.pi * 4258 * dt * np.reshape(rfp_rfse_am,
                                                                   (np.size(
                                                                       rfp_rfse_am),
                                                                    1)))
        Mxyrfse = 2 * np.conj(a) * b
        mse_rfse = np.sqrt(np.mean((abs(Mxyrfse)-ideal_prof)**2))
        print('TB = {}, MSE = {}'.format(tb_rfse, mse_rfse))
        tb_rfse += tb_interval
    bsse_durs.append(dt*np.size(rfp_rfse_am))  # seconds
    # pl.LinePlot(Mxyrfse, title='Final RFSE Mxy')


pyplot.plot(pbc, rfse_durs)
pyplot.plot(pbc, bsse_durs)
pyplot.show()





