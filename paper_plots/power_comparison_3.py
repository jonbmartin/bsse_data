import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

dt = 2e-6
pbc_min = 2
pbc_max = 8  # actually sims down to 1 G below (7G)
pbw_min = 0.1
pbw_max = 1.0
pbc = np.arange(pbc_min, pbc_max, 0.5)  # gauss, b1 range to sim over
pbw_list = np.arange(pbw_min, pbw_max, 0.1)
pwr_ratio_list = []
mse_ratio_list = []
t_bs = []
t_rfse = []
d1, d2 = 0.01, 0.01

sim_max = 8
nb1 = 1000
db1 = sim_max / nb1
b1_sim = np.linspace(0, sim_max, nb1)

def predict_multiphoton_res(om1, om2, n):
    '''

    :param om1: wbs (khz), typically + for bsse
    :param om2: wbs (khz), typically - for bsse
    :param n: multiphoton resonance of interest
    :return:b1_mp, resonance in G
    '''
    av = 0.5*(om1 + om2)
    res_cond_num = 1/n
    pbc_hz = (abs(om1-av)/res_cond_num-av)*1000
    om1 = om1 * 1000
    b1_mp = (om1/4258)*np.sqrt((1+(pbc_hz)/om1)**2-1)
    return b1_mp

tb_start = 2
tb_interval = 0.05
err_thresh = 0.04
bs_offsets = [10000, 15000, 20000]

# PART 1 - sweep across PBC, hold PBW
pbw = 0.25
bsse_durs = []
bsse_pwrs = []# sweep from 0.5G PBC to 6G PBC, at 0.3G PBW, error

rfse_durs = []
rfse_pwrs = []
new_pbc_sim = True
if new_pbc_sim:
    for bs_ii in list(bs_offsets):
        for pb in list(pbc):
            tb_bsse, tb_rfse = tb_start, tb_start
            rmse_bs = np.inf
            print('##########################################')
            print('PBC = {} G, PBW = {} G'.format(pb, pbw))
            print('##########################################')
            print('BSSE pulse, offset = {}:'.format(bs_ii))

            # create ROI for BSSE - should exclude MP res. Will also apply to
            # RFSE to make the comparison fair
            w = np.ones(np.size(b1_sim))

            # predict the multiphoton resonance
            wrf = rf.b12wbs(bs_offset=bs_ii, b1=pb)  # Hz
            bmp = predict_multiphoton_res(bs_ii / 1000, -wrf / 1000, 3)  # G
            w[int(bmp/db1-pbw/db1/2):int(bmp/db1+pbw/db1/2)] = 0


            while rmse_bs > err_thresh:

                bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb_bsse, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                        pbc=[pb], d1e=d1, d2e=d2, rampfilt=True, bs_offset=bs_ii)
                print('BSSE length = {} ms'.format(dt*np.size(bsrf)*1000))
                rfp_bs = bsrf + rfp_ex
                ideal_prof = np.zeros(np.size(b1_sim))
                ideal_prof[int(pb / db1 - pbw / db1 / 2):int(
                    pb / db1 + pbw / db1 / 2)] = 1

                a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                                  np.array([[1]]), 0, b1_sim.reshape(len(b1_sim), 1))
                Mxybs = 2 * np.conj(a) * b
                rmse_bs = np.sqrt(np.mean((abs(Mxybs)*w - ideal_prof*w) ** 2))
                print('TB = {}, MSE = {}'.format(tb_bsse, rmse_bs))
                tb_bsse += tb_interval

            bsse_durs.append(dt*np.size(bsrf))  # seconds
            bsse_pwrs.append(np.sum(abs(rfp_bs)**2)*dt)
            pl.LinePlot(Mxybs,title='Final BSSE Mxy')

            # only do this once
            print('RFSE pulse:')
            rmse_rfse = np.inf
            while rmse_rfse > err_thresh:
                if bs_ii != 10000:
                    break

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
                rmse_rfse = np.sqrt(np.mean((abs(Mxyrfse)*w - ideal_prof*w) ** 2))
                print('TB = {}, MSE = {}'.format(tb_rfse, rmse_rfse))
                tb_rfse += tb_interval
            # pl.LinePlot(Mxyrfse, title='Final RFSE Mxy')
            rfse_durs.append(dt*np.size(rfp_rfse_am))  # seconds
            rfse_pwrs.append(np.sum(abs(rfp_rfse_am)**2)*dt)
    sio.savemat('data/pulse_comparison_3_data_0d25pbw_tol0d12.mat', {'pbc': pbc, 'rfse_durs': rfse_durs,
                                                  'bsse_durs': bsse_durs,
                                                  'rfse_pwrs': rfse_pwrs,
                                                  'bsse_pwrs': bsse_pwrs})

# plotting: should plot slice (0.25), [1 6], in between(0.625) [1.5 6] slab (1.0) [2 6]
##########################################3
# data plotting - PBC 0d25
# sweep from 0.5G PBC to 7G PBC, at 0.25G PBW
# error thresh = 0.12 rmse

powerlim = [0.5, 2.6]
durlim = [0.000, 0.020]

fig, axs = pyplot.subplots(2, 3)
data = sio.loadmat('data/pulse_comparison_data_0d25pbw_tol0d12.mat')
nb1 = 13
pbc = data['pbc'][:,0:nb1]
rfse_durs = data['rfse_durs']
bsse_durs = data['bsse_durs']
rfse_pwrs = data['rfse_pwrs']
bsse_pwrs = data['bsse_pwrs']

# duration plotting
dur20, dur15, dur10 = bsse_durs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_durs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_durs[:, 0:nb1].T
axs[0, 0].plot(pbc.T, dur20, '#6D3AAB')
axs[0, 0].plot(pbc.T, dur15, '#B250E8')
axs[0, 0].plot(pbc.T, dur10, '#D39BE8')

axs[0, 0].plot(pbc.T, rfse_durs[:,:nb1].T,'#000000')
axs[0, 0].set_ylabel('pulse dur (s)')
axs[0, 0].set_title('Slice excitation (0.25 G)')
axs[0, 0].set_ylim(durlim)

# power plotting
pwr20, pwr15, pwr10 = bsse_pwrs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_pwrs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_pwrs[:, 0:nb1].T
axs[1, 0].plot(pbc.T, pwr20/rfse_pwrs[:,:nb1].T,'#A03010')
axs[1, 0].plot(pbc.T, pwr15/rfse_pwrs[:,:nb1].T,'#C75A3A')
axs[1, 0].plot(pbc.T, pwr10/rfse_pwrs[:,:nb1].T,'#E48D74')

axs[1, 0].plot(pbc.T, np.ones(np.size(rfse_pwrs[:,:nb1])),'#000000',linestyle='--')
axs[1, 0].set_ylabel('relative power')
axs[1, 0].set_xlabel('PBC (G)')
axs[1, 0].set_ylim(powerlim)

########################################333
# data plotting - PBC 0d625
# sweep from 1.5G PBC to 7G PBC, at 0.625G PBW, error thresh 0.12 rmse
data = sio.loadmat('data/pulse_comparison_data_0d625pbw_tol0d12.mat')
nb1 = 12
pbc = data['pbc'][:,0:nb1]
rfse_durs = data['rfse_durs']
bsse_durs = data['bsse_durs']
rfse_pwrs = data['rfse_pwrs']
bsse_pwrs = data['bsse_pwrs']
# duration plotting
dur20, dur15, dur10 = bsse_durs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_durs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_durs[:, 0:nb1].T
axs[0, 1].plot(pbc.T, dur20, '#6D3AAB')
axs[0, 1].plot(pbc.T, dur15, '#B250E8')
axs[0, 1].plot(pbc.T, dur10, '#D39BE8')

axs[0,1].plot(pbc.T, rfse_durs[:,:nb1].T,'#000000')
axs[0,1].set_ylim(durlim)
axs[0, 1].set_title('Thick slice excitation (0.625 G)')



# power plotting
pwr20, pwr15, pwr10 = bsse_pwrs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_pwrs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_pwrs[:, 0:nb1].T
axs[1, 1].plot(pbc.T, pwr20/rfse_pwrs[:,:nb1].T,'#A03010')
axs[1, 1].plot(pbc.T, pwr15/rfse_pwrs[:,:nb1].T,'#C75A3A')
axs[1, 1].plot(pbc.T, pwr10/rfse_pwrs[:,:nb1].T,'#E48D74')

axs[1, 1].plot(pbc.T, np.ones(np.size(rfse_pwrs[:,:nb1])),'#000000',linestyle='--')
axs[1, 1].set_xlabel('PBC (G)')
axs[1, 1].set_ylim(powerlim)

########################################333
# data plotting - PBC 1d0
# sweep from 1.5G PBC to 7G PBC, at 0.625G PBW, error thresh 0.12 rmse
data = sio.loadmat('data/pulse_comparison_data_1d0pbw_tol0d12.mat')
nb1 = 11
pbc = data['pbc'][:,0:nb1]
rfse_durs = data['rfse_durs']
bsse_durs = data['bsse_durs']
rfse_pwrs = data['rfse_pwrs']
bsse_pwrs = data['bsse_pwrs']
# duration plotting
dur20, dur15, dur10 = bsse_durs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_durs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_durs[:, 0:nb1].T
axs[0, 2].plot(pbc.T, dur20, '#6D3AAB')
axs[0, 2].plot(pbc.T, dur15, '#B250E8')
axs[0, 2].plot(pbc.T, dur10, '#D39BE8')

axs[0, 2].plot(pbc.T, rfse_durs[:,:nb1].T,'#000000')
axs[0, 2].legend(['BSSE-20 kHz offset','BSSE-15kHz offset', 'BSSE-10kHz offset','RFSE'])
axs[0, 2].set_ylim(durlim)
axs[0, 2].set_title('Slab excitation (1.0 G)')



# power plotting
pwr20, pwr15, pwr10 = bsse_pwrs[:, (2*nb1+2):(3*nb1+2)].T,\
                      bsse_pwrs[:, (nb1+1):(2*nb1+1)].T,\
                      bsse_pwrs[:, 0:nb1].T
axs[1, 2].plot(pbc.T, pwr20/rfse_pwrs[:,:nb1].T,'#A03010')
axs[1, 2].plot(pbc.T, pwr15/rfse_pwrs[:,:nb1].T,'#C75A3A')
axs[1, 2].plot(pbc.T, pwr10/rfse_pwrs[:,:nb1].T,'#E48D74')

axs[1, 2].plot(pbc.T, np.ones(np.size(rfse_pwrs[:,:nb1])),'#000000',linestyle='--')
axs[1, 2].legend(['BSSE-20 kHz offset','BSSE-15kHz offset', 'BSSE-10kHz offset'])
axs[1, 2].set_xlabel('PBC (G)')
axs[1, 2].set_ylim(powerlim)

pyplot.show()

