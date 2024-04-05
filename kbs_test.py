import sigpy.mri.rf as rf
import sigpy.plot as pl
import numpy as np

am = np.ones((8000,1))
fm = np.ones((8000,1)) * 4000  #  Hz
dt = 1e-6

kbs = rf.calc_kbs(am,fm,0.008)
print(kbs)
