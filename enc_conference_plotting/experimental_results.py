import numpy as np
import sys
import scipy.io as sio
sys.path.append(r"/Users/guille/Dropbox/code/sigpy_staging/sigpy-rf")
import sigpy.plot as pl
import matplotlib.pyplot as pyplot

sim_data = sio.loadmat('sim_1khz.mat')
exp_data = sio.loadmat('exp_results_best.mat')
b1 = exp_data['b1_sqrt_range']
exp = exp_data['dynamics_1G']
sim = sim_data['sim_1khz']
# pl.LinePlot(exp_data['dynamics_1G'].T)
# pl.LinePlot(sim_data['sim_1khz'])

scalefact = 6.7
b1_fine = np.linspace(0,b1[0,33],np.size(sim))
pyplot.figure()
pyplot.plot(b1_fine,scalefact*abs(sim).T,'#1021a1',)
pyplot.plot(b1[:,0:33].T,exp, 'b--')
pyplot.xlabel('$B_1$ (Gauss)')
pyplot.ylabel('$|M_{xy}|$ (a.u.)')
pyplot.legend(['Simulated','Experimental'])
pyplot.yticks([0,1,2,3])
pyplot.show()
