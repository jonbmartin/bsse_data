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

target, sens = problem_2d(32)
pl.ImagePlot(sens)
pl.ImagePlot(target)
dim = target.shape[0]
g, k1, t, s = rf.spiral_arch(1.5, 0.175, 4e-6, 200, 50)
k1 = np.flipud(k1)
pyplot.plot(k1[:,0])
pyplot.plot(k1[:,1])
pyplot.show()

# A = rf.PtxSpatialExplicit(sens, k1, dt=4e-6, img_shape=target.shape,
#                           b0=None)
# pulses = sp.mri.rf.stspa(target, sens, st=None, coord=k1, dt=4e-6,
#                          max_iter=60, alpha=0.001, tol=1E-4,
#                          phase_update_interval=200, explicit=True)
#
# pl.LinePlot(pulses)
# pl.ImagePlot(A*pulses, colormap='jet')


# 1 coil BS
coord = np.expand_dims(k1[:,0],0) # just 1 dimensional frequency modulation
sens = sens[0,:,:] # just 1 coil

dt = 1e-6
db1 = 0.01
pbc = 1.5
b1min = 0
b1max = 2.5
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

pbw = 0.5 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 4
bs_offset = 5000

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)
t = np.linspace(0, np.size(rfp_ss),np.size(rfp_ss))*dt*1000
pyplot.plot(t, np.squeeze(abs(rfp_bs+rfp_ss)))
pyplot.title('conventional BSSE')
pyplot.xlabel('t (ms)')
pyplot.show()
full_pulse = rfp_bs + rfp_ss
a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.title('Bloch Sim')
pyplot.ylabel('|Mxy|')
pyplot.xlabel('B1+ (Gauss)')
pyplot.ylim([0,1.05])
pyplot.xlim([b1min,b1max])
pyplot.show()

# pl.LinePlot(rfp_ss)
print(f'pulse duration = {np.size(rfp_ss)*dt*1000} ms')
coord = np.ones(np.shape(rfp_ss))*2  # bs trajectory
coord = np.expand_dims(np.linspace(0,0.0000001,np.size(coord)),0)
wrf = np.ones(np.size(rfp_ss))*(1/5000)
wrf[0:591]=0
wrf[-591:]=0
fm_bs = np.diff(np.unwrap(np.imag(np.log(rfp_bs / abs(rfp_bs)))))/(dt*2*np.pi)  # Hz
fm_bs = np.concatenate((fm_bs,np.zeros((1,1))), axis =1)
coord = np.cumsum(wrf)*dt
pl.LinePlot(coord)
coord = np.expand_dims(coord,0)
b = np.ones(np.shape(coord)).T
Abs = rf.PtxBSExplicit(sens, coord, dt=dt, img_shape=np.shape(sens),b1scale=8)

m = Abs@rfp_ss.T
pl.ImagePlot(np.reshape(m,np.shape(sens)))
print(coord)
