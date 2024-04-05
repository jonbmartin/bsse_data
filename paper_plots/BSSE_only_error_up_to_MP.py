import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
import scipy.io as sio

from matplotlib import cm
from collections import OrderedDict

cmaps = OrderedDict()


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

dt = 4e-6
pbc_min = 1  # originally 1
pbc_max = 10  # actually sims down to 1 G below (7G) # originally 8
pbc_step = 0.25
pbw_min = 0.1 # originally 0.1
pbw_max = 1.5 # originally 1.5
pbw_step = 0.05
pbc_list = np.arange(pbc_min, pbc_max, pbc_step)  # gauss, b1 range to sim over
pbw_list = np.arange(pbw_min, pbw_max, pbw_step)
pwr_ratio_list = []
mse_ratio_list = []
t_bs = []
t_rfse = []

sim_max = 10
db1 = 0.02
# nb1 = 1200  # was failing to get large enough passband at 1000
# db1 = sim_max / nb1
# b1_sim = np.linspace(0, sim_max, nb1)

tb = 6
bs_start = 5000 # Hz
d1, d2 = 0.01, 0.01

# PART 1 - sweep across PBC, hold PBW
bsse_durs_v = []
bsse_pwrs_v = []# sweep from 0.5G PBC to 6G PBC, at 0.3G PBW, error
bsse_off_v = []
rfse_durs_v = []
rfse_pwrs_v = []
rfse_rmse_v = []
bsse_rmse_v = []
rfse_stoprip_v = []
bsse_stoprip_v = []
rfse_passrip_v = []
bsse_passrip_v = []
bsse_mp3_v = []

newsim = True
if newsim:
    for pbc in list(pbc_list):
        for pbw in list(pbw_list):

            print('##########################################')
            print('PBC = {} G, PBW = {} G'.format(pbc, pbw))
            print('##########################################')

            #BSSE pulse design
            rmse_bs = 1000  # way above anything realistic
            bsse_dur = 0  # in s - way above anything realistic
            bsse_off = 7500
            bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                             pbc=[pbc], d1e=d1, d2e=d2, rampfilt=True, bs_offset=bsse_off)
            bsse_dur = dt*np.size(bsrf)
            print('Equal duration for BSSE = {} Hz off'.format(bsse_off))
            # predict the multiphoton resonance
            wrf = rf.b12wbs(bs_offset=bsse_off, b1=pbc)  # Hz
            bmp = predict_multiphoton_res(bsse_off / 1000, -wrf / 1000, 3)  # G

            db1 = 0.01
            # nb1 = 1200  # was failing to get large enough passband at 1000
            # db1 = sim_max / nb1
            max_b1 = np.minimum(sim_max,bmp-pbw) # simulate to 10G or to 1st mp
            b1_sim = np.linspace(0, max_b1, int(max_b1/db1))

            ideal_prof = np.zeros(np.size(b1_sim))
            ideal_prof[int(pbc / db1 - pbw / db1 / 2):int(pbc / db1 + pbw / db1 / 2)] = 1


            print('BSSE offset = {}, length = {} ms'.format(bsse_off, dt*np.size(bsrf)*1000))
            rfp_bs = bsrf + rfp_ex
            a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                              np.array([[1]]), 0, b1_sim.reshape(len(b1_sim), 1))
            Mxybs = np.squeeze(2 * np.conj(a) * b)
            import sigpy.plot as pl
            # pl.LinePlot(Mxybs)

            # create a ROI for RMSE & BSSE - should exclude transition regions and MP
            w = np.ones(np.size(b1_sim))

            # exclude the transition region
            ftw_g = rf.dinf(d1, d2) / tb * pbw
            w[int((pbc - pbw / 2) / db1 - ftw_g / db1 ):int((pbc - pbw / 2) / db1 + ftw_g / db1)] = 0
            w[int((pbc + pbw / 2) / db1 - ftw_g / db1):int((pbc + pbw / 2) / db1 + ftw_g / db1)] = 0
            # exclude multiphoton
            w[int(bmp/db1-pbw/db1/2):int(bmp/db1+pbw/db1/2)] = 0
            # make a copy for w_stopband before proceeding

            # # TODO: BODGEY FIX. EXPAND the DONT CARE REGION
            # w[int((pbc - pbw / 1.5) / db1 - ftw_g / db1 / 2):int((pbc - pbw / 1.5) / db1 + ftw_g / db1 / 2)] = 0
            # w[int((pbc + pbw / 1.5) / db1 - ftw_g / db1 / 2):int((pbc + pbw / 1.5) / db1 + ftw_g / db1 / 2)] = 0


            w_stopband = w
            w_stopband[int((pbc - pbw / 2) / db1):int((pbc + pbw / 2) / db1)] = 0
            # pyplot.plot(b1_sim,abs(Mxyrfse))
            # pyplot.plot(b1_sim,w)
            # pyplot.title("RFSE, PBC = {}, PBW = {}".format(pbc,pbw))
            # pyplot.show()



            # pyplot.plot(b1_sim, abs((Mxyrfse).T))
            # pyplot.plot(b1_sim, ideal_prof*w)
            # pyplot.plot(b1_sim, w)
            # pyplot.show()
            # pyplot.plot(b1_sim, abs((Mxybs).T))
            # pyplot.plot(b1_sim, ideal_prof*w)
            # pyplot.plot(b1_sim, w)
            # pyplot.show()

            # compare RMSE error
            # TODO: BODGEY FIX FOR THE BSSE SHIFT. Manual shift 1 db1
            # w_bsse = w
            # w_bsse[int((pbc - pbw / 1.5) / db1 - ftw_g / db1 / 2):int((pbc - pbw / 1.5) / db1 + ftw_g / db1 / 2)] = 0
            # w_bsse[int((pbc + pbw / 1.5) / db1 - ftw_g / db1 / 2):int((pbc + pbw / 1.5) / db1 + ftw_g / db1 / 2)] = 0
            # if (pbw == 0.1 or pbw == 0.15) and pbc>=3:
            #     w_bsse = np.roll(w_bsse,1)
            # if (pbw == 0.2 or pbw == 0.25 or pbw ==0.3) and pbc >= 5.4:
            #     w_bsse = np.roll(w_bsse, 1)
            # pyplot.plot(b1_sim,abs(Mxybs))
            # pyplot.plot(b1_sim,w)
            # pyplot.title("BSSE, PBC = {}, PBW = {}".format(pbc,pbw))
            # pyplot.show()
            rmse_bs = np.sqrt(np.mean((abs(Mxybs)*w - ideal_prof*w) ** 2))

            print('BSSE RMSE = {}'.format(rmse_bs))
            print('---')

            # compare stopband ripple
            max_stop_ripple_bsse = np.max(abs(Mxybs)*w_stopband)

            print('BSSE Max Stopband Ripple = {}'.format(max_stop_ripple_bsse))
            print('---')

            #compare passband ripple
            w_passband = np.zeros(np.size(w))
            w_passband[int((pbc - pbw / 2) / db1 + ftw_g / db1 / 2):int((pbc + pbw / 2) / db1 - ftw_g / db1 / 2)] = 1
            w_passband_inds=np.nonzero(w_passband)
            max_pass_ripple_rfse = 0
            max_pass_ripple_bsse = 0
            # max_pass_ripple_rfse = np.max(abs(Mxyrfse[w_passband_inds]))-np.min(abs(Mxyrfse[w_passband_inds]))
            # max_pass_ripple_bsse = np.max(abs(Mxybs[w_passband_inds]))-np.min(abs(Mxybs[w_passband_inds]))
            print('RFSE Max Passband Ripple = {}'.format(max_pass_ripple_rfse))
            print('BSSE Max Passband Ripple = {}'.format(max_pass_ripple_bsse))
            print('---')

            bsse_durs_v.append(dt*np.size(bsrf))  # seconds
            bsse_pwrs_v.append(np.sum(abs(rfp_bs)**2)*dt)
            bsse_off_v.append(bsse_off)
            bsse_rmse_v.append(rmse_bs)
            bsse_passrip_v.append(max_pass_ripple_bsse)
            bsse_stoprip_v.append(max_stop_ripple_bsse)
            bsse_mp3_v.append(bmp)
            # pl.LinePlot(Mxybs,title='Final BSSE Mxy')

    dic = {'rfse_durs_v':rfse_durs_v, 'rfse_pwrs_v':rfse_pwrs_v,
           'rfse_rmse_v':rfse_rmse_v,'rfse_passrip_v':rfse_passrip_v,
           'rfse_stoprip_v':rfse_stoprip_v, 'bsse_durs_v':bsse_durs_v,
           'bsse_pwrs_v':bsse_pwrs_v, 'bssse_off_v':bsse_off_v,
           'bsse_rmse_v':bsse_rmse_v, 'bsse_passrip_v':bsse_passrip_v,
           'bsse_stoprip_v':bsse_stoprip_v, 'bsse_mp3_v':bsse_mp3_v}
    sio.savemat('bsse_error_data.mat',dic)
# sio.savemat('data/pulse_comparison_data_0d625pbw_tol0d12.mat', {'pbc': pbc, 'rfse_durs': rfse_durs,
#                                               'bsse_durs': bsse_durs,
#                                               'rfse_pwrs': rfse_pwrs,
#                                               'bsse_pwrs': bsse_pwrs})

# plotting: should plot slice (0.25), [1 6], in between(0.625) [1.5 6] slab (1.0) [2 6]
##########################################3
# data plotting - PBC 0d25
# sweep from 0.5G PBC to 7G PBC, at 0.25G PBW
# error thresh = 0.12 rmse

data = sio.loadmat('bsse_error_data.mat')
y = len(pbw_list)
x = len(pbc_list)
bsse_rmse_v = np.reshape(data['bsse_rmse_v'], (x,y))
rfse_rmse_v = np.reshape(data['rfse_rmse_v'],(x,y))
bsse_stoprip_v = np.reshape(data['bsse_stoprip_v'], (x,y))
rfse_stoprip_v = np.reshape(data['rfse_stoprip_v'], (x,y))
bsse_durs_v = np.reshape(data['bsse_durs_v'], (x,y))
bsse_pwrs_v = np.reshape(data['bsse_pwrs_v'], (x,y))
rfse_pwrs_v = np.reshape(data['rfse_pwrs_v'], (x,y))
bsse_off_v = np.reshape(data['bssse_off_v'],(x,y))
bsse_mp3_v = np.reshape(data['bsse_mp3_v'],(x,y))



# 26 PBC, 18 PBW
# plot max ripples, rmse, bs offsets, power ratio, time
# Max ripples bsse, rmse bsse, and each of those for BSSE alone

fig, axs = pyplot.subplots(2, 2,sharex=True,sharey=True)
max_pbw_index=22
max_pbc_index = 32
aspect = 0.1
extent=[pbw_min, pbw_step*max_pbw_index,pbc_step*max_pbc_index,pbc_min]

# marker symbol
im0 = axs[0, 0].imshow(bsse_rmse_v[0:max_pbc_index, 0:max_pbw_index], cmap ='inferno', vmin=0, vmax=0.02, extent=extent,aspect=aspect)
axs[0, 0].set_title("BSSE RMSE")
fig.colorbar(im0, ax=axs[0, 0])
axs[0,0].set_ylabel('PBC (G)')

im1 = axs[0, 1].imshow(bsse_stoprip_v[0:max_pbc_index, 0:max_pbw_index], cmap ='inferno', vmin=0, vmax=0.05, extent=extent, aspect=aspect)
axs[0, 1].set_title("BSSE Max Ripple")
fig.colorbar(im1, ax=axs[0,1])

im2 = axs[1, 0].imshow(bsse_rmse_v[0:max_pbc_index, 0:max_pbw_index]/rfse_rmse_v[0:max_pbc_index, 0:max_pbw_index],
                       cmap='jet', vmin=0, vmax=5, extent=extent, aspect=aspect)
axs[1, 0].set_title("BSSE RMSE / RFSE RMSE")
fig.colorbar(im2, ax=axs[1,0])
axs[1,0].set_xlabel('PBW (G)')
axs[1,0].set_ylabel('PBC (G)')

im3 = axs[1, 1].imshow(bsse_stoprip_v[0:max_pbc_index, 0:max_pbw_index]/rfse_stoprip_v[0:max_pbc_index, 0:max_pbw_index],
                       cmap='jet', vmin=0, vmax=5, extent=extent, aspect=aspect)
axs[1, 1].set_title("BSSE Max Rip. / RFSE Max. Rip")
fig.colorbar(im3, ax=axs[1,1])
axs[1,1].set_xlabel('PBW (G)')


pyplot.tight_layout()
fig = pyplot.gcf()
fig.set_size_inches(7, 10)
pyplot.show()
aspect=0.1

fig, axs = pyplot.subplots(1,4, sharex=False, sharey=True)
im0 = axs[0].imshow(bsse_off_v[0:max_pbc_index, 0:max_pbw_index], cmap='jet',
                    vmin=0, vmax=50000, extent=extent,aspect=aspect)
axs[0].set_title("$\omega_{off}$ (Hz)")
axs[0].set_xlabel('PBW (G)')
axs[0].set_ylabel('PBC (G)')
fig.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(bsse_durs_v[0:max_pbc_index, 0:max_pbw_index], cmap='jet',
                    vmin=0.002, vmax=0.020, extent=extent,aspect=aspect)
axs[1].set_title("Pulse Duration (ms)")
axs[1].set_xlabel('PBW (G)')

fig.colorbar(im1, ax=axs[1])
im2 = axs[2].imshow(bsse_pwrs_v[0:max_pbc_index, 0:max_pbw_index]/rfse_pwrs_v[0:max_pbc_index, 0:max_pbw_index],
                    vmin=0, vmax=1, extent=extent,cmap='jet',aspect=aspect)
axs[2].set_title("$P_{BSSE}/P_{RFSE}$")
axs[2].set_xlabel('PBW (G)')

fig.colorbar(im2, ax=axs[2])

im3 = axs[3].imshow(bsse_mp3_v[0:max_pbc_index, 0:max_pbw_index],
                    vmin=2, vmax=25, extent=extent,cmap='jet',aspect=aspect)
axs[3].set_title("$B_{1,N=3}$")
axs[3].set_xlabel('PBW (G)')

fig.colorbar(im3, ax=axs[3])

# pyplot.tight_layout()
fig.set_size_inches(9*1.2, 4*1.2)

pyplot.show()

