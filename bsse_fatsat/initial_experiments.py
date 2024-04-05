import scipy.io as sio
import numpy as np
import sigpy.plot as pl
import matplotlib.pyplot as pyplot
import sigpy.mri.rf as rf

dataset = 'chardat'
if dataset =='chardat':
    b1 = sio.loadmat('data/char_b1s.mat')
    mask = b1['mask']
    b1 = b1['b1']
    slice_ind = 25
    mask_slice = mask[:,:,slice_ind]
    test_slice = np.transpose(b1[:, :, slice_ind, :], (2, 0, 1))
    datasize = 80
elif dataset == 'zhipeng':
    # Zhipeng's maps
    b1 = sio.loadmat('data/b1data.mat')
    mask_slice = b1['mask'].astype(int)
    test_slice = np.transpose(b1['B1p'],(2,0,1))  # Nc X Y
    datasize = 64
elif dataset == 'ax8':
    b1 = sio.loadmat('data/LoopTx8_Head1_B1p_Axial.mat')
    mask = b1['mask']
    b1 = b1['b1']
    slice_ind = 25
    test_slice = np.transpose(b1[:, :, slice_ind, :], (2, 0, 1))
    mask_slice = mask[:,:,slice_ind]
    datasize = 10


b1max_scale = 4  # Gauss
test_slice *= b1max_scale

pl.ImagePlot(test_slice)
pl.ImagePlot((1-mask_slice)*test_slice)

# design a b1 selective pulse

dt = 4e-6
db1 = 0.01
pbc = 1.85
pbw = 0.5 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 4
bs_offset = 5000

b1min = 0
b1max = pbc+1
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=bs_offset)
full_pulse = rfp_bs + rfp_ss

print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

pyplot.plot(b1, np.squeeze(abs(Mxyfull)))
pyplot.title('excitation profile in B1+')
pyplot.show()

# simulate the pulse across the entire head
b1_head = np.squeeze(test_slice.flatten())
a_head, b_head = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1_head.reshape(np.size(b1_head), 1))

Mxyfull_head = 2 * np.conj(a_head) * b_head
Mxyfull_head = np.reshape(Mxyfull_head, (8, datasize, datasize))

pl.ImagePlot(np.sum(Mxyfull_head,0))