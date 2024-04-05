import numpy as np
import sigpy.mri.rf as rf
import sigpy.plot as pl


fm = rf.adiabatic_bs_fm(n=512, dur=2e-3, b1p=2., k=20.,
                     gamma=2*np.pi*42.58)

pl.LinePlot(np.split(fm,2)[0])

