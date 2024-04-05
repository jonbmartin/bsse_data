import numpy as np
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf
import scipy.io as sio

# WINNING DESIGN
n_offres = 180 * 1
input = np.repeat([[0, 0, 1]], n_offres, axis=0)
pbc = 1.4
pbw = 0.3
dt_ex = 4e-6  # sz
gam = 4258
tb = 8
offset = 7500
d1e = 0.001
d2e = 0.001
b1sim = np.arange(pbc, pbc+0.025, 0.02)  # JBM originall 0.025G res

newsim = True

if newsim:
    # DESIGN THE EXCITATION PULSE
    rfp_bs, rfp, _ = rf.dz_bssel_rf(dt=dt_ex, tb=tb, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                            pbc=[pbc], d1e=d1e, d2e=d2e,
                            rampfilt=True, bs_offset=offset)

    full_ex = rfp + rfp_bs
    pulse_dur = np.size(full_ex)*dt_ex
    b1 = np.arange(0, 3, 0.02)  # gauss, b1 range to sim over
    a, b = rf.abrm_hp(
        2 * np.pi * 4258 * dt_ex * full_ex.reshape((1, np.size(full_ex))),
        np.zeros(np.size(full_ex)),
        np.array([[1]]), 0, b1.reshape(np.size(b1), 1))

    Mxyfull = 2 * np.conj(a) * b
    pyplot.figure()
    pyplot.plot(b1, np.abs(Mxyfull.transpose()))
    pyplot.ylabel('|Mxy|')
    pyplot.xlabel('Gauss')
    pyplot.title('Excitation')
    pyplot.show()

    # DESIGN THE REFOCUSING PULSE AND REWINDER
    rfp_bs_inv, rfp_inv, rw180 = rf.dz_bssel_rf(dt=dt_ex, tb=tb, ndes=128,
                                                ptype='se', flip=np.pi,
                                                pbw=pbw, pbc=[pbc], d1e=d1e,
                                                d2e=d2e, rampfilt=False,
                                                bs_offset=offset)
    pyplot.plot(np.angle(rfp.T))
    pyplot.plot(np.angle(rfp_inv.T))
    dur_inv = dt_ex*np.size(rfp_inv)  # s
    dur_rw = dt_ex*np.size(rw180)  # s

    # rfp_inv *= np.exp(1j * np.ones(np.size(rfp_bs_inv))*np.pi)

    pyplot.show()

    print('Inversion pulse length = {} ms'.format(np.size(rfp_inv)*dt_ex*1000))

    # SIMULATE REFOCUSING EFFICIENCY
    full_bs = rfp_bs_inv + rfp_inv
    a180, b180 = rf.abrm_hp(2 * np.pi * 4258 * dt_ex * full_bs,
                            np.zeros(np.size(rfp_bs_inv.T)),
                            np.array([[1]]), 0, b1.reshape(len(b1), 1))

    print('Max refocusing efficiency = {}'.format(np.max(abs(np.real(b180.T**2)))))
    pyplot.figure()
    pyplot.title('BSSE Beta Squared')
    # pyplot.plot(b1, np.real(b180.T ** 2), label='BSSE Re(beta**2)')
    # pyplot.plot(b1, np.imag(b180.T ** 2), label='BSSE Im(beta**2)')
    pyplot.plot(b1, np.abs(b180.T) ** 2, label = 'BSSE |(beta)|**2')
    pyplot.legend(loc='lower right')
    pyplot.xlabel('Gauss')
    # pyplot.ylim([-1, 1])
    pyplot.show()

    print('Inversion duration = {} s'.format(dt_ex*np.size(rfp_bs_inv)))
    # simulate their behavior across b1

    #### PULSE SEQUENCE ####
    f0 = np.linspace(-180, 180, n_offres)  # inhomogeneity
    t1 = np.infty   # s, formerly inf
    t2 = np.infty  # s, brain gray matter
    dt = 1e-3  # s
    te = 0.015  # s
    n_te = 8

    all_decay_b1 = np.array([])
    rw180 = np.expand_dims(rw180, 0)

    for bb in range(np.size(b1sim)):
        b1_iter = b1sim[bb]
        print('B1 Being Simulated = {} G'.format(b1_iter))
        # ADD REWINDERS TO REFOCUSING PULSES AND SCALE
        bsse_ex = (rfp_bs + rfp) * gam * 2 * np.pi * dt_ex * b1_iter
        bsse_inv = np.concatenate((rw180, rfp_bs_inv + rfp_inv, rw180),
                                  1) * gam * 2 * np.pi * dt_ex * b1_iter

        ### 90 EX
        output = mr.bloch_forward(input, np.squeeze(bsse_ex), f0, t1, t2, dt_ex)
        print('90 ex')

        te_2 = np.zeros(1)
        n_ex = int(te // dt) - int(pulse_dur/dt)  # samples to wait before starting inv

        decay = np.zeros((n_offres, 3, n_ex))
        for ii in range(0, n_ex):
            output = mr.bloch_forward(np.real(output), te_2, f0, t1, t2, dt)
            decay[:, :, ii] = output

        # 180 train
        for ii in range(n_te):
            print('te {}'.format(ii))
            # apply the 180 every 2*tau
            rfp_inv_flip = rfp_inv
            # code to alternate the inversion btwn -y and y
            if ii % 2 == 0:
                rfp_inv_flip = rfp_inv * np.exp(-1j * np.ones(np.size(rfp_bs_inv)) * np.pi/2)
            else:
                rfp_inv_flip = rfp_inv

            bsse_inv = np.concatenate((rw180, rfp_bs_inv + rfp_inv_flip, rw180),
                                      1) * gam * 2 * np.pi * dt_ex * b1_iter

            output = mr.bloch_forward(np.real(output), np.squeeze(bsse_inv), f0, t1, t2, dt_ex)
            te_2 = np.zeros(1)
            n_tau = int(te // dt) - int(pulse_dur/dt/2)

            npts = 2*n_tau
            decay2 = np.zeros((n_offres, 3, npts), dtype=np.complex)
            # wait for the delay, sim each time point
            for ii in range(0, npts):
                output = mr.bloch_forward(np.real(output), te_2, f0, t1, t2, dt)
                decay2[:, :, ii] = output

            decay = np.concatenate((decay, decay2), axis=2)

        all_decay_summed = np.sum(decay, axis=0)  # sum across f0
        sequence_mxy = np.abs(np.sqrt(all_decay_summed[1, :]**2+all_decay_summed[0, :]**2))/n_offres
        #pl.LinePlot(sequence_mxy)
        all_decay_b1 = np.vstack([all_decay_b1, sequence_mxy]) if all_decay_b1.size else sequence_mxy

    slice_mxy = np.sum(all_decay_b1, 0)
    pl.LinePlot(slice_mxy)
    sio.savemat('cpmg_data.mat', {'slice_mxy': slice_mxy,
                                  'inv_rf': rfp_bs_inv + rfp_inv,
                                  'dt':dt,
                                  'rw': rw180,
                                  'ex_rf': rfp_bs + rfp,
                                  'beta_cplx': b180})

## plotting the figure
vars = sio.loadmat('cpmg_data.mat')

#load data
rf_vec = np.hstack((vars['ex_rf'],np.zeros(np.shape(vars['ex_rf'])),
                    np.zeros(np.shape(vars['ex_rf'])),vars['rw'],
                    vars['inv_rf'],vars['rw'],np.zeros(np.shape(vars['ex_rf']))))

fig = pyplot.figure(constrained_layout=True)

gs1 = fig.add_gridspec(nrows=2, ncols=4, wspace=0.1)
ax1 = fig.add_subplot(gs1[0, :])
ax1.plot(np.abs(rf_vec.T),'k')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_ylabel('|RF|', rotation=0, labelpad=15)


ax2 = fig.add_subplot(gs1[1, 0:2])
b1 = np.arange(0, 3, 0.02)  # gauss, b1 range to sim over
# ax2.plot(b1, np.real(vars['beta_cplx'].T ** 2))
# ax2.plot(b1, np.imag(vars['beta_cplx'].T ** 2))
ax2.plot(b1,np.abs(vars['beta_cplx'].T**2))
ax2.set_xlabel(r'$B_1$ (G)')
ax2.set_title(r'Refocusing profile $|\beta^2|$')
# ax2.legend((r'Re($\beta^2$)', r'Im($\beta^2$)'))

ax3 = fig.add_subplot(gs1[1, 2:4])
t = np.arange(0, np.size(vars['slice_mxy'])*vars['dt'], vars['dt'])
ax3.plot(t, vars['slice_mxy'].T)
ax3.set_xlabel('t (s)')
ax3.set_title('integrated slice signal (a.u.)')

# fig.patch.set_visible(False)
# fig.tight_layout()
#pyplot.show()
pyplot.savefig('figures_out/cpmg_fullpbw.png',dpi=300)
pyplot.show()
