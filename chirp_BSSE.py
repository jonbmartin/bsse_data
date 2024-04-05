import numpy as np
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri.rf as rf
from scipy.signal import chirp

dt=1e-6
b1 = np.arange(0.0, 2.0, 0.02)  # gauss, b1 range to sim over

rfp_bs, rfp_ss = rf.dz_bssel_chirp_rf(dt=dt, T=0.005, pbb=0.5, pbt=1.0, bs_offset=10000)
rfp_bs += rfp_ss
pl.LinePlot(rfp_bs)


a, b = rf.abrm_hp(2 * np.pi * 4258 * dt * (rfp_bs).reshape((1, len(rfp_bs.T))), np.zeros(len(rfp_bs.T)),
                  np.array([[1]]), 0, b1.reshape(len(b1), 1))
Mxybs = 2 * np.conj(a) * b
