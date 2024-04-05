import numpy as np
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as pyplot

n_offres = 100
input = np.repeat([[0, 0, 1]], n_offres, axis=0)
b1_90 =  np.pi / 2 * np.ones(1000) / 1000  #90x
pl.LinePlot(b1_90)
b1_180 =1j* np.pi  * np.ones(1000) / 1000  #180y


dt_ex = 1e-6  # s
f0 = np.linspace(-np.pi, np.pi, n_offres)  # inhomogeneity
t1 = np.infty  # s
t2 = np.infty  # s, brain white matter
output = mr.bloch_forward(input, b1_90, f0, t1, t2, dt_ex)
print('90 ex')

dt =1e-3
te_2 = np.zeros(1)
npts = 4000
decay = np.zeros((n_offres,3,npts))
for ii in range(0,npts):
    output = mr.bloch_forward(np.real(output),te_2,f0,t1,t2,dt)
    decay[:,:,ii]=output

# pl.LinePlot(np.sum(decay,axis=0))
# apply 180
print('te_2')
output = mr.bloch_forward(np.real(output), b1_180, f0, t1, t2, dt_ex)
print('180')

te_2 = np.zeros(1)
npts = 8000
decay2 = np.zeros((n_offres,3,npts))
for ii in range(0,npts):
    output = mr.bloch_forward(np.real(output),te_2,f0,t1,t2,dt)
    decay2[:,:,ii]=output
# pl.LinePlot(np.sum(decay2,axis=0))

all_decay = np.concatenate((decay,decay2),axis=2)
all_decay_summed = np.sum(all_decay,axis=0)
# pl.LinePlot(np.sum(all_decay,axis=0))
pyplot.figure()
pyplot.plot(np.abs(all_decay_summed[1,:]))
pyplot.show()


# simulate across off-resonances
# >> > input = np.repeat([[0, 0, 1]], 100, axis=0)
# >> > b1 = np.pi / 2 * np.ones(1000) / 1000
# >> > dt = 1
# >> > f0 = np.linspace(-np.pi, np.pi, 100)
# >> > t1 = np.full(100, np.infty)
# >> > t2 = np.full(100, np.infty)
# >> > output = bloch_forward(input, b1, f0, t1, t2, dt)