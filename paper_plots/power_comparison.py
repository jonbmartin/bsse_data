import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl

dt = 2e-6
pbc_min = 2
pbc_max = 6
pbw_min = 0.4
pbw_max = 1.2
pbc = np.arange(pbc_min, pbc_max, 1.0)  # gauss, b1 range to sim over
pbw_list = np.arange(pbw_min, pbw_max, 0.2)
pwr_ratio_list = []
mse_ratio_list = []
t_bs = []
t_rfse = []

tb = 6
bs_offset = 15000

for pb in list(pbc):
    for pbw in list(pbw_list):
        print('PBC = {} G, PBW = {} G'.format(pb, pbw))

        bsrf, rfp_ex, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                pbc=[pb], d1e=0.01, d2e=0.01, rampfilt=True, bs_offset=bs_offset)
        print('BSSE pulse length = {} ms'.format(dt*np.size(bsrf)*1000))
        rfp_bs = bsrf + rfp_ex
        # #pl.LinePlot(rfp_bs)
        b1_sim = np.linspace(pb-2*pbw, pb+2*pbw, 100)
        db1 = (pb+2*pbw - (pb-2*pbw))/100
        ideal_prof = np.zeros(np.shape(b1_sim))
        ideal_prof[38:62] = 1
        a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                          np.array([[1]]), 0, b1_sim.reshape(len(b1_sim), 1))
        Mxybs = 2 * np.conj(a) * b
        mse_bs = np.sqrt(np.mean((abs(Mxybs)-ideal_prof)**2))
        print(mse_bs)

        pyplot.plot(abs(Mxybs).T)
        pyplot.plot(np.expand_dims(ideal_prof, 0).T)
        pyplot.title('BSSE')
        pyplot.show()
        # # b1_sim_range = [pbc-pbw, pbc+pbw]

        [rfp_rfse_am, rfp_rfse_fm] = rf.dz_b1_rf(dt=dt, tb=tb, ptype='ex', flip=np.pi/2, pbw=pbw, pbc=pb,
                                               d1=0.01, d2=0.01, os=8, split_and_reflect=True)

        rfse_rf = rfp_rfse_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(rfp_rfse_fm))

        [a, b] = rf.sim.abrm_nd(2 * np.pi * dt * rfp_rfse_fm, np.reshape(b1_sim, (np.size(b1_sim),1)),
                                2 * np.pi * 4258 * dt * np.reshape(rfp_rfse_am,
                                                                   (np.size(
                                                                       rfp_rfse_am),
                                                                    1)))
        Mxyrfse = -2 * np.real(a * b) + 1j * np.imag(np.conj(a) ** 2 - b ** 2)
        mse_rfse = np.sqrt(np.mean((abs(Mxyrfse)-ideal_prof)**2))
        print(mse_rfse)

        pyplot.plot(abs(Mxyrfse).T)
        pyplot.plot(np.expand_dims(ideal_prof,0).T)
        pyplot.title('RFSE')
        pyplot.show()

        mse_ratio = mse_bs / mse_rfse
        mse_ratio_list.append(mse_ratio)

        pwr_bs = np.sum(abs(rfp_bs) ** 2) / np.size(rfp_bs)
        pwr_rfse = np.sum(abs(rfp_rfse_am) ** 2) / np.size(rfp_rfse_am)
        pwr_ratio = pwr_bs / pwr_rfse
        pwr_ratio_list.append(pwr_ratio)
        t_bs.append(dt*np.size(rfp_bs))
        t_rfse.append(dt*np.size(rfp_rfse_am))

fig, (ax1, ax2) = pyplot.subplots(figsize=(13, 2), ncols=2)

# display power
print
pwrfig = ax1.imshow(np.reshape(np.array(mse_ratio_list), (len(pbc), len(pbw_list))), cmap='magma',extent=[pbw_min, pbw_max, pbc_max, pbc_min])
ax1.set_title(r'$RMSE_{bs} / RMSE_{rfse}$')
ax1.set_xlabel('PBW (G)')
ax1.set_ylabel('PBC (G)')
ax1.set_aspect('auto')

pyplot.colorbar(pwrfig, ax=ax1)

# display duration ratio
timefig = ax2.imshow(np.reshape(np.array(t_bs)/np.array(t_rfse), (len(pbc),len(pbw_list))), cmap='RdBu', vmin=0, vmax=2,extent=[pbw_min, pbw_max, pbc_max, pbc_min])
ax2.set_title(r'$T_{bs} / T_{rfse}$')
ax2.set_xlabel('PBW (G)')
ax2.set_ylabel('PBC (G)')
ax2.set_aspect('auto')

pyplot.colorbar(timefig, ax=ax2)
pyplot.show()


