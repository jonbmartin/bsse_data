import sigpy.mri.rf as rf
import sigpy as sp
import numpy as np
import scipy.ndimage.filters as filt
from sigpy.mri import rf, linop, sim
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import scipy.integrate as integrate

def problem_2d(dim):
    img_shape = [dim, dim]
    sens_shape = [8, dim, dim]

    # target - slightly blurred circle
    x, y = np.ogrid[-img_shape[0] / 2: img_shape[0] - img_shape[0] / 2,
           -img_shape[1] / 2: img_shape[1] - img_shape[1] / 2]
    circle = x * x + y * y <= int(img_shape[0] / 6) ** 2
    target = np.zeros(img_shape)
    target[circle] = 1
    target = filt.gaussian_filter(target, 1)
    target = target.astype(np.complex)

    sens = sim.birdcage_maps(sens_shape)

    return target, sens

dim = 32
target, sens = problem_2d(dim)
pl.ImagePlot(sens)
pl.ImagePlot(target)
dim = target.shape[0]
g, k1, t, s = rf.spiral_arch(1.5, 0.175, 4e-6, 200, 50)
k1 = np.flipud(k1)
pyplot.plot(k1[:,0])
pyplot.plot(k1[:,1])
pyplot.show()

A = rf.PtxSpatialExplicit(sens, k1, dt=4e-6, img_shape=target.shape,
                          b0=None)
# pulses = sp.mri.rf.stspa(target, sens, st=None, coord=k1, dt=4e-6,
#                          max_iter=60, alpha=0.001, tol=1E-4,
#                          phase_update_interval=200, explicit=True)
#
# pl.LinePlot(pulses)
# pl.ImagePlot(A*pulses, colormap='jet')


# 1 coil BS
coord = np.expand_dims(k1[:,0],0) # just 1 dimensional frequency modulation
# sens = sens[0,:,:] # just 1 coil

dt = 1e-6
db1 = 0.01
pbc = 3
b1min = 0
b1max = pbc+1
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

pbw = 0.5 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 4
bs_offset = 4000

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)

t = np.linspace(0, np.size(rfp_ss),np.size(rfp_ss))*dt*1000

# pl.LinePlot(rfp_ss)
print(f'pulse duration = {np.size(rfp_ss)*dt*1000} ms')
b1_traj = np.ones((8, np.size(rfp_bs)))*0.01
sens_onecoil = np.expand_dims(sens[0,:,:],0)
kb1 = np.cumsum(b1_traj**2, axis=1)
for ii in range(8):
    kb1[ii,:] = kb1[ii,:] * np.sin(kb1[ii,:]*30)
    kb1[ii,:] += -np.min(kb1[ii,:])
kb1 /= 10
pl.LinePlot(kb1)
# Abs = rf.PtxBSAMExplicit(sens, bs_offset, kb1, dt, img_shape=[dim, dim])
Abs = rf.PtxGradAndBSAMExplicit(sens, bs_offset, kb1, dt, img_shape=[dim, dim], coord=coord)
# Abs_explicit = rf.PtxBSAMExplicit(sens, bs_offset, kb1, dt, img_shape=[dim, dim], ret_array=True)

I = sp.linop.Identity((8, coord.shape[0]))
b = Abs.H *  target
alpha = 0
rf = np.tile(rfp_ss,(8,1))
rf = np.zeros(np.shape(rf)) + 0j * np.zeros(np.shape(rf))

alg_method = sp.alg.ConjugateGradient(Abs.H * Abs,
                                      b, rf, P=None,
                                      max_iter=300, tol=1e-6)
while not alg_method.done():
    alg_method.update()
    print(f'iter # {alg_method.iter}')
rf = np.tile(rfp_ss,(8,1))
m = Abs*alg_method.x
# A_sample = Abs_explicit[:,0:3200]
# m_explicit = A_sample @ alg_method.x[0,:]
pl.LinePlot(alg_method.x)
pyplot.plot(t, abs(alg_method.x.T))
pyplot.title('8 channel |b_ex(t)| RF')
pyplot.xlabel('t (ms)')
pyplot.show()
pl.ImagePlot(m, title='excitation')
# pl.ImagePlot(np.reshape(m_explicit,[32, 32]))
pl.ImagePlot(abs(m-target), title='error')
print(coord)
