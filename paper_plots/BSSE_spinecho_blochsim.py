import numpy as np
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf

n_offres = 100
input = np.repeat([[0, 0, 1]], n_offres, axis=0)
pbc = 3
pbw=0.55
dt_ex = 2e-6  # s
gam = 4258

b1_sim = np.arange(0,4,0.05)
[rfp_rfse_am, rfp_rfse_fm] = rf.dz_b1_rf(dt=2e-6, tb=6, ptype='se',
                                         flip=np.pi, pbw=pbw, pbc=pbc,
                                         d1=0.01, d2=0.01, os=8,
                                         split_and_reflect=True)

rfse_rf = rfp_rfse_am * np.exp(1j * dt_ex * 2 * np.pi * np.cumsum(rfp_rfse_fm))

[a, b] = rf.sim.abrm_nd(2 * np.pi * dt_ex * rfp_rfse_fm,
                        np.reshape(b1_sim, (np.size(b1_sim), 1)),
                        2 * np.pi * 4258 * dt_ex * np.reshape(rfp_rfse_am,
                                                           (np.size(
                                                               rfp_rfse_am),
                                                            1)))

print(
    'Max refocusing efficiency = {}'.format(np.max(abs(np.real(b.T ** 2)))))
pyplot.figure()
pyplot.title('BSSE Beta Squared')
pyplot.plot(b1_sim, np.real(b.T ** 2), label='BSSE Re(beta**2)')
pyplot.plot(b1_sim, np.imag(b.T ** 2), label='BSSE Im(beta**2)')
pyplot.plot(b1_sim, np.abs(b.T**2))
pyplot.legend(loc='lower right')
pyplot.xlabel('Gauss')
pyplot.ylim([-1, 1])
pyplot.show()
b1_strength = 3  # gauss

rfp_bs, rfp, _ = rf.dz_bssel_rf(dt=dt_ex, tb=2, ndes=128, ptype='ex', flip=np.pi / 2, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=7500)
bsse_ex = (rfp_bs + rfp)*gam*2*np.pi*dt_ex*b1_strength
pl.LinePlot(bsse_ex, title='BSSE 90 Degree Pulse')

rfp_bs_inv, rfp_inv, _ = rf.dz_bssel_rf(dt=dt_ex, tb=2, ndes=128, ptype='se', flip=np.pi, pbw=pbw,
                        pbc=[pbc], d1e=0.01, d2e=0.01,
                        rampfilt=False, bs_offset=7500)
bsse_inv = (rfp_bs_inv + rfp_inv)*gam*2*np.pi*dt_ex*b1_strength
pl.LinePlot(bsse_inv, title='BSSE 180 Degree Pulse')


print('Inversion duration = {} s'.format(dt_ex*np.size(bsse_inv)))
# simulate their behavior across b1
b1 = np.arange(0, 4, 0.02)  # gauss, b1 range to sim over

a90, b90 = rf.abrm_hp(2 * np.pi * 4258 * dt_ex * (rfp_bs+rfp), np.zeros(np.size(bsse_ex.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
a180, b180 = rf.abrm_hp(2 * np.pi * 4258 * dt_ex * (rfp_bs_inv+rfp_inv), np.zeros(np.size(bsse_inv.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))

pyplot.figure()
pyplot.title('BSSE Beta Squared')
pyplot.plot(b1, np.real(b180.T**2), label ='BSSE Re(beta**2)')
pyplot.plot(b1, np.imag(b180.T**2), label ='BSSE Im(beta**2)')
pyplot.legend(loc = 'upper left')
pyplot.xlabel('Gauss')
pyplot.show()




f0 = np.linspace(-np.pi, np.pi, n_offres)  # inhomogeneity
t1 = np.inf  # s, formerly inf
t2 = np.inf  # s, brain white matter
output = mr.bloch_forward(input, np.squeeze(bsse_ex), f0, t1, t2, dt_ex)
print('90 ex')

dt =1e-3
te_2 = np.zeros(1)
npts = 2000
decay = np.zeros((n_offres,3,npts))
for ii in range(0,npts):
    output = mr.bloch_forward(np.real(output),te_2,f0,t1,t2,dt)
    decay[:,:,ii]=output

# pl.LinePlot(np.sum(decay,axis=0))
# apply 180
print('te_2')
output = mr.bloch_forward(np.real(output), np.squeeze(bsse_inv), f0, t1, t2, dt_ex)
print('180')

te_2 = np.zeros(1)
npts = 4000
decay2 = np.zeros((n_offres,3,npts))
for ii in range(0,npts):
    output = mr.bloch_forward(np.real(output),te_2,f0,t1,t2,dt)
    decay2[:,:,ii]=output
# pl.LinePlot(np.sum(decay2,axis=0))

all_decay = np.concatenate((decay,decay2),axis=2)
all_decay_summed = np.sum(all_decay,axis=0)
# pl.LinePlot(np.sum(all_decay,axis=0))

# plotting RF time points ... no need to do this
rf_data = np.zeros(6000)
rf_data[0]=1
rf_data[2000]=2


fig, ax1 = pyplot.subplots()
color = 'tab:red'
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('|Mxy|', color=color)
ax1.plot(np.abs(all_decay_summed[1,:])/n_offres, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0,1])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('RF', color=color)  # we already handled the x-label with ax1
ax2.plot(rf_data, color=color)
ax2.tick_params(axis='y', right=False)
ax2.set_yticklabels([])
ax2.set_ylim([0,1])

fig.tight_layout()
pyplot.show()


# simulate across off-resonances
# >> > input = np.repeat([[0, 0, 1]], 100, axis=0)
# >> > b1 = np.pi / 2 * np.ones(1000) / 1000
# >> > dt = 1
# >> > f0 = np.linspace(-np.pi, np.pi, 100)
# >> > t1 = np.full(100, np.infty)
# >> > t2 = np.full(100, np.infty)
# >> > output = bloch_forward(input, b1, f0, t1, t2, dt)