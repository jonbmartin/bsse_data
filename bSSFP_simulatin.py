import numpy as np
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf
import scipy.io as sio

n_offres = 360 * 1
input = np.repeat([[0, 0, 1]], n_offres, axis=0)
pbc = 1.4
pbw = 0.3
dt_ex = 4e-6  # sz
gam = 4258
tb = 4
offset = 7500
d1e = 0.001
d2e = 0.001
b1sim = np.array([pbc, pbc-0.3])

newsim = True
use_refined_pulse = True

if newsim:
    # DESIGN THE EXCITATION PULSE
    flip_angle = 120
    pulse_samples = 25
    b1scale = flip_angle/(gam*360*pulse_samples*dt_ex)
    rf_pulse_base = np.ones([1, pulse_samples], dtype=np.complex) * b1scale  # gauss
    pulse_dur = np.size(rf_pulse_base)*dt_ex


    #### PULSE SEQUENCE ####
    f0 = np.linspace(-3000, 3000, n_offres)  # inhomogeneity
    t1 = 0.85   # s, formerly inf
    t2 = 0.07  # s, brain gray matter
    dt = 1e-4  # s
    te = 0.002  # s
    n_te = 250

    all_decay_b1 = np.array([])

    # ADD REWINDERS TO REFOCUSING PULSES AND SCALE

    ### 90 EX
    output = mr.bloch_forward(input, np.squeeze(rf_pulse_base), f0, t1, t2, dt_ex)
    print('90 ex')

    te_2 = np.zeros(1)
    n_ex = int(te // dt) - int(pulse_dur/dt)  # samples to wait before starting inv

    decay = np.zeros((n_offres, 3, n_ex))
    for ii in range(0, n_ex):
        output = mr.bloch_forward(np.real(output), te_2, f0, t1, t2, dt)
        decay[:, :, ii] = output

    # 180 train
    for ii in range(n_te):
        rf_pulse = rf_pulse_base
        if np.mod(ii,2) == 0:
            rf_pulse *= -1 #np.exp(-1j * np.ones(np.size(rf_pulse)) * np.pi)

        output = mr.bloch_forward(np.real(output), np.squeeze(rf_pulse), f0, t1, t2, dt_ex)
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

# decay dimensions: (F0, M, Time)
decay_steady_state = np.sum(decay[:,:,np.shape(decay)[2]-1000:np.shape(decay)[2]],2)/n_offres
pyplot.plot(f0,np.sqrt(decay_steady_state[:,0]**2 + decay_steady_state[:,1]**2))
pyplot.xlabel('off-resonance (Hz)')
pyplot.ylabel('signal (a.u.)')
pyplot.show()

pl.LinePlot(all_decay_summed/n_offres)
all_decay_summed_mxy = np.sqrt(all_decay_summed[0,:]**2 + all_decay_summed[1,:]**2)
t = np.linspace(0,dt*np.size(all_decay_summed_mxy),np.size(all_decay_summed_mxy))

pyplot.plot(t, abs(all_decay_summed_mxy)/n_offres)
pyplot.ylabel('|Mxy|/M0')
pyplot.xlabel('time (s)')
pyplot.title('signal timecourse')
pyplot.show()
## plotting the figure
# vars = sio.loadmat('cpmg_data.mat')
#
# #load data
# rf_vec = np.hstack((vars['ex_rf'],np.zeros(np.shape(vars['ex_rf'])),
#                     np.zeros(np.shape(vars['ex_rf'])),vars['rw'],
#                     vars['inv_rf'],vars['rw'],np.zeros(np.shape(vars['ex_rf']))))

fig = pyplot.figure(constrained_layout=True, figsize=[6,4])

gs1 = fig.add_gridspec(nrows=2, ncols=4, wspace=0.1)
ax1 = fig.add_subplot(gs1[0, :])
ax1.plot(np.abs(rf_pulse.T),'k')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_ylabel('|RF|', rotation=0, labelpad=15)


# ax2 = fig.add_subplot(gs1[1, 0:2])
# db1 = 0.02
# b1 = np.arange(pbc-0.5, pbc+0.5, db1)  # gauss, b1 range to sim over
# beta_refined = np.abs(vars['beta_cplx'].T**2)
# beta_refined = beta_refined[int((pbc-0.5)//db1):int((pbc+0.5)//db1),:]
# b180_unrefined = b180_unrefined[:, int((pbc-0.5)//db1):int((pbc+0.5)//db1)]
# # ax2.plot(b1, np.real(vars['beta_cplx'].T ** 2))
# # ax2.plot(b1, np.imag(vars['beta_cplx'].T ** 2))
# ax2.plot(b1/10,np.abs(b180_unrefined.T**2), color='red')
# ax2.plot(b1/10,beta_refined, color='#1f77b4')
# ax2.legend([ 'unrefined pulse', 'refined pulse'])
# ax2.set_xlabel(r'$B_1^+$ (mT)')
# ax2.set_title(r'Refocusing profile $|\beta^2|$')
# ax2.legend((r'Re($\beta^2$)', r'Im($\beta^2$)'))

ax3 = fig.add_subplot(gs1[1, 2:4])
# b1_dat = vars['all_decay_b1']
t = np.arange(0, np.size(vars['slice_mxy'])*vars['dt'], vars['dt'])
ax3.plot(t, b1_dat[0,:].T)
ax3.plot(t, b1_dat[1,:].T)
ax3.set_xlabel('t (s)')
ax3.set_title('integrated signal (a.u.)')
ax3.legend([r'$B_1^+$=0.14 mT',r'$B_1^+$=0.17 mT'],loc=7)

# fig.patch.set_visible(False)
# fig.tight_layout()
#pyplot.show()
# pyplot.savefig('figures_out/cpmg_oneb1.png',dpi=300)
# pyplot.show()
