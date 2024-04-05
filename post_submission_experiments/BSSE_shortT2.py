import numpy as np
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf
import scipy.io as sio

n_offres =  360 * 1
# input = np.repeat([[0, 0, 1]], n_offres, axis=0)
input = np.array([0, 0, 1])
pbc = 1.4
pbw = 0.3
dt_ex = 4e-6  # sz
gam = 4258
tb = 4
offset = 20000
d1e = 0.001
d2e = 0.001
# b1sim = np.array([pbc, pbc+0.3])
b1sim = np.arange(0, 3, 0.1)  # gauss, b1 range to sim over

newsim = True
use_refined_pulse = False

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
    # pyplot.plot(np.angle(rfp.T))
    # pyplot.plot(np.angle(rfp_inv.T))
    dur_inv = dt_ex*np.size(rfp_inv)  # s
    dur_rw = dt_ex*np.size(rw180)  # s

    # rfp_inv *= np.exp(1j * np.ones(np.size(rfp_bs_inv))*np.pi)

    pyplot.show()

    print('Inversion pulse length = {} ms'.format(np.size(rfp_inv)*dt_ex*1000))

    # SIMULATE REFOCUSING EFFICIENCY
    if use_refined_pulse:
        refined_data = sio.loadmat('after_refine_50_pbc1d4pbw0d3.mat')
        full_bs = refined_data['pulse_ref']
        full_bs /= (np.max(full_bs)*0.915)
        unrefined_bs = rfp_bs_inv + rfp_inv
    else:
        full_bs = rfp_bs_inv + rfp_inv
        unrefined_bs = rfp_bs_inv + rfp_inv

    pyplot.figure()
    pyplot.plot(abs(full_bs.T))
    pyplot.plot(abs(rfp_bs_inv+rfp_inv).T)
    pyplot.legend(['refined', 'unrefined'])
    pyplot.show()

    a180_unrefined, b180_unrefined = rf.abrm_hp(2 * np.pi * 4258 * dt_ex * unrefined_bs,
                            np.zeros(np.size(rfp_bs_inv.T)),
                            np.array([[1]]), 0, b1.reshape(len(b1), 1))

    a180, b180 = rf.abrm_hp(2 * np.pi * 4258 * dt_ex * full_bs,
                            np.zeros(np.size(full_bs.T)),
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
    # f0 = np.linspace(-180, 180, n_offres)  # inhomogeneity
    f0 = 0
    t1 = np.infty   # s, formerly inf
    t2 = 50e-3  # s, macromolecule
    dt = 1e-4  # s
    te = 0.020  # s
    n_te = 8

    all_decay_b1 = np.array([])
    rw180 = np.expand_dims(rw180, 0)
    output_mag = []
    for bb in range(np.size(b1sim)):
        b1_iter = b1sim[bb]
        print('B1 Being Simulated = {} G'.format(b1_iter))
        # ADD REWINDERS TO REFOCUSING PULSES AND SCALE
        bsse_ex = (rfp_bs + rfp) * gam * 2 * np.pi * dt_ex * b1_iter
        bsse_inv = np.concatenate((rw180, rfp_bs_inv + rfp_inv, rw180),
                                  1) * gam * 2 * np.pi * dt_ex * b1_iter

        ### 90 EX
        output = mr.bloch_forward(input, np.squeeze(bsse_ex), f0, t1, t2, dt_ex)
        output_mag.append(output)
        print(np.sum(output,axis=0))
        print('90 ex')
    output_mag = np.array(output_mag).T
    pl.LinePlot(output_mag)


