import scipy.io as sio
import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import ImageGrid

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


b1data = sio.loadmat('b1map.mat')
exdata = sio.loadmat('dataEX.mat')

slice_to_show=1
b1map = b1data['b1map']+0.3
dataBASE = exdata['dataBASE']
dataSHIFT = exdata['dataSHIFT']
dataFA = exdata['dataFA']
dataTB = exdata['dataTB']

dataHP_F = np.abs(b1data['fftdataneg'])
dataHP_full = b1data['fftdataneg']
# pl.ImagePlot(dataFA)


# apply previous crop to hard pulse image. Crop to just slices with signal
dataHP_F = dataHP_F[0:60,9:49,:]
dataHP_full = dataHP_full[0:60,9:49,:]
dataBASE = dataBASE[0:60,9:49,:]
dataSHIFT = dataSHIFT[0:60,9:49,:]
dataTB = dataTB[0:60,9:49,:]
dataFA = dataFA[0:60,9:49,:]


# now crop all down to area around phantom
bot, top = 10, 35
dataHP_F = dataHP_F[bot:top,:,0:2]
dataHP_full = dataHP_full[bot:top,:,0:2]
b1map = b1map[bot:top,:,0:2]
dataBASE = dataBASE[bot:top,:,0:2]
dataSHIFT = dataSHIFT[bot:top,:,0:2]
dataTB = dataTB[bot:top,:,0:2]
dataFA = dataFA[bot:top,:,0:2]

# 1D b1map - to create labels
b1map_1d = np.flipud(abs(np.mean(b1map[5:18,:,1],0))) # b1 map, collected to 1D
gridpoints = np.linspace(0,35,8).astype('int')  # grid locations we want to label
b1_labels = list(trunc(b1map_1d[gridpoints],2))
b1_labels.insert(0, 1.9)
# pl.LinePlot(b1map_1d)


# apply hp
# dataBASE/=np.sqrt(dataHP_F)
# dataSHIFT/=np.sqrt(dataHP_F)
# dataTB/=np.sqrt(dataHP_F)
# dataFA/=np.sqrt(dataHP_F)

# apply b1 map to get rid of transmit sensitivities
# dataHP_F/=b1map
dataBASE/=b1map
dataSHIFT/=b1map
dataTB/=b1map
dataFA/=b1map

# pl.ImagePlot(dataHP_F[:,:,slice_to_show])


mask = dataHP_F > np.max(dataHP_F)*0.20
# pl.ImagePlot(mask[:,:,slice_to_show])
b1map*=mask
dataHP_F*=mask
dataHP_full*=mask
dataBASE*=mask
dataSHIFT*=mask
dataFA*=mask
dataTB*=mask

dataHP_full = np.sum(dataHP_full,2)
dataHP_F = np.sum(dataHP_F,2)
dataBASE = np.sum(dataBASE,2)
dataFA = np.sum(dataFA,2)
dataSHIFT = np.sum(dataSHIFT,2)
dataTB = np.sum(dataTB,2)


# dataHP_F= dataHP_f[:mask
# pl.ImagePlot(b1map[:,:,slice_to_show])
# pl.ImagePlot(dataBASE[:,:,slice_to_show])
# pl.ImagePlot(dataSHIFT[:,:,slice_to_show])
# pl.LinePlot(np.mean(abs(dataBASE),0))
# pyplot.figure()
# # pyplot.plot(np.flipud(b1map_1d),np.mean(abs(dataBASE),0))
# # pyplot.plot(np.flipud(b1map_1d), np.mean(abs(dataFA),0))
# # pyplot.show()


fig = pyplot.figure()
grid = ImageGrid(fig, 111,
                nrows_ncols = (2,5),
                axes_pad = 0.25,
                cbar_location = "right",
                cbar_mode="edge",
                cbar_size="5%",
                cbar_pad=0.05
                )

# fig, axs = pyplot.subplots(nrows=1,ncols=5,sharex=True,sharey=True,figsize=(12,6))
# axs[0].imshow(abs(np.flipud(dataHP_F[:,:,slice_to_show].T)))
dims = np.shape(dataTB)
dx = 2 # mm
dy = 3.5
vmin = 0
vmax = 7.5E8
vmin_phs = -3.14
vmax_phs=3.14
im4=grid[4].imshow(abs(np.flipud(b1map[:,:,slice_to_show].T))/10, cmap='jet')
grid[4].set_yticklabels(('0', '0', '17.5', '35', '52.5', '70','87.5', '105', '122.5'))

grid[0].set_xticklabels(('test','0','10','20'))
# grid[0].set_xticklabels(('test','0','30'))

# grid[0].set_yticklabels(('0', '0', '17.5', '35', '52.5', '70','87.5', '105', '122.5'))
# grid[0].set_yticklabels(('0', '0', '35', '70', '105'))
grid[0].set_yticklabels(('0', '0.192', '0.189', '0.171', '0.153', '0.137','0.123', '0.108', '0.087'))
grid[5].set_yticklabels(('0', '0.192', '0.189', '0.171', '0.153', '0.137','0.123', '0.108', '0.087'))


# grid[0].set_yticklabels(b1_labels)
im0=grid[0].imshow(abs(np.flipud(dataBASE[:,:].T)), cmap='gray',vmin=vmin, vmax=vmax)
im1=grid[1].imshow(abs(np.flipud(dataSHIFT[:,:].T)), cmap='gray',vmin=vmin, vmax=vmax)
im2=grid[2].imshow(abs(np.flipud(dataFA[:,:].T)), cmap='gray',vmin=vmin, vmax=vmax)
im3=grid[3].imshow(abs(np.flipud(dataTB[:,:].T)), cmap='gray',vmin=vmin, vmax=vmax)

phsBASE = np.angle(np.flipud(dataBASE[:,:].T)/np.flipud(dataHP_full[:,:].T))
phsBASE = np.nan_to_num(phsBASE)
# phsBASE = np.unwrap(np.nan_to_num(phsBASE), axis=0)
phsSHIFT = np.angle(np.flipud(dataSHIFT[:,:].T)/np.flipud(dataHP_full[:,:].T))
phsSHIFT = np.nan_to_num(phsSHIFT)
phsFA = np.angle(np.flipud(dataFA[:,:].T)/np.flipud(dataHP_full[:,:].T))
phsFA = np.nan_to_num(phsFA)
phsTB = np.angle(np.flipud(dataTB[:,:].T)/np.flipud(dataHP_full[:,:].T))
phsTB = np.nan_to_num(phsTB)
phs_cmap = 'RdBu'
phs0=grid[5].imshow(phsBASE, cmap=phs_cmap,vmin=vmin_phs, vmax=vmax_phs)
phs1=grid[6].imshow(phsSHIFT, cmap=phs_cmap,vmin=vmin_phs, vmax=vmax_phs)
phs2=grid[7].imshow(phsFA, cmap=phs_cmap,vmin=vmin_phs, vmax=vmax_phs)
phs3=grid[8].imshow(phsTB, cmap=phs_cmap,vmin=vmin_phs, vmax=vmax_phs)


# grid[4].title.set_text('$B_1^+$ map (mT)')
grid[4].title.set_text(u"\u2220"+r" $M_{xy}$ (rad)")
grid[0].title.set_text('Base Pulse')
grid[1].title.set_text(r'PBC $\rightarrow$ 0.11 mT')
grid[2].title.set_text(r'FA $\rightarrow$ 45$^{\circ}$')
grid[3].title.set_text(r'TB $\rightarrow$ 1.5')
grid[7].set_xlabel('relative z position (mm)')
grid[0].set_ylabel('spatially gridded $B_1^+$ (mT)')
grid[5].set_ylabel('spatially gridded $B_1^+$ (mT)')
grid[9].axis("off")
pyplot.colorbar(im4, cax=grid.cbar_axes[0])
cbarphs= pyplot.colorbar(phs3, cax=grid.cbar_axes[1])
pyplot.show()

import sigpy.plot as pl
pl.LinePlot(np.unwrap(phsTB,axis=0).T)



# 1D Figure

base_d = np.mean(abs(dataBASE),0)
shift_d = np.mean(abs(dataSHIFT),0)
tb_d = np.mean(abs(dataTB),0)
fa_d = np.mean(abs(dataFA),0)

pbw = 0.3
d1 = 0.01
d2 = 0.01
dt = 1e-6
bs_offset =7500

b1_ax = np.flipud(b1map_1d)
b1_sim = np.arange(np.min(b1map_1d), np.max(b1map_1d), 0.01)
# base pulse
tb=4
pbc=1.4

rfp_bs_base, rfp_ss_base, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_base = rfp_bs_base+rfp_ss_base

a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_base.reshape((1, np.size(rfp_base))), np.zeros(np.size(rfp_base)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxybase = 2 * np.conj(a) * b

# FA pulse
tb=4
pbc=1.4
rfp_bs_fa, rfp_ss_fa, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_fa = rfp_bs_fa+rfp_ss_fa
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_fa.reshape((1, np.size(rfp_fa))), np.zeros(np.size(rfp_fa)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxyfa = 2 * np.conj(a) * b

# shift pulse
tb=4
pbc=1.1
rfp_bs_shift, rfp_ss_shift, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_shift = rfp_bs_shift+rfp_ss_shift
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_shift.reshape((1, np.size(rfp_shift))), np.zeros(np.size(rfp_shift)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxyshift = 2 * np.conj(a) * b

# tb pulse
tb=1.5
pbc=1.4
rfp_bs_tb, rfp_ss_tb, _ = rf.dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='ex', flip=np.pi / 2, pbw=pbw,
                                   pbc=[pbc], d1e=d1, d2e=d2,
                                   rampfilt=True, bs_offset=7500)
rfp_tb = rfp_bs_tb+rfp_ss_tb
a, b = rf.abrm_hp(2*np.pi*4258*dt*rfp_tb.reshape((1, np.size(rfp_tb))), np.zeros(np.size(rfp_tb)),
                    np.array([[1]]), 0, b1_sim.reshape(np.size(b1_sim), 1))
Mxytb = 2 * np.conj(a) * b


fig, axs = pyplot.subplots(nrows=3,ncols=2,sharex=True,sharey=False,figsize=(10,5))

experimental_ymax = 4.15E8
sim_ymax = 1.15 # gets the 90s roughly even
# shifting plots

# convert b1 axis to mT
b1_ax /= 10
b1_sim /= 10

axs[0,0].set_title('h) Simulated Profile')
axs[0,1].set_title('i) Experimental Profile')
axs[1,0].set_ylabel(r'$|M_{xy}|/M_0$')
axs[1,1].set_ylabel(r'signal (a.u.)')
axs[2,0].set_xlabel(r'$B_1^+$ (mT)')
axs[2,1].set_xlabel(r'$B_1^+$ (mT)')



axs[2,0].plot(b1_sim.T,abs(Mxybase.T))
axs[2,0].plot(b1_sim.T,abs(Mxyshift.T))
axs[2,1].plot(b1_ax.T,base_d.T)
axs[2,1].plot(b1_ax.T,shift_d.T)
axs[2,0].set_ylim([0,sim_ymax])
axs[2,1].set_ylim([0,experimental_ymax])
# axs[2,1].legend([r"PBC=1.4G",r"PBC=1.1G"],loc='center left',bbox_to_anchor=(1,0.5))
axs[2,1].legend(["PBC=\n0.14 mT","PBC=\n0.11 mT"],loc='upper left')


# TB plots
axs[1,0].plot(b1_sim.T,abs(Mxybase.T))
axs[1,0].plot(b1_sim.T,abs(Mxytb.T),'r')
axs[1,1].plot(b1_ax.T,base_d.T)
axs[1,1].plot(b1_ax.T,tb_d.T,'r')
axs[1,0].set_ylim([0,sim_ymax])
axs[1,1].set_ylim([0,experimental_ymax])
#axs[1,1].legend([r'TB=4',r'TB=1.5'],loc='center left',bbox_to_anchor=(1,0.5))
axs[1,1].legend([r'TB=4',r'TB=1.5'],loc='upper left')


# FA plots
axs[0,0].plot(b1_sim.T,abs(Mxybase.T))
axs[0,0].plot(b1_sim.T,abs(Mxyfa.T),'g')
axs[0,1].plot(b1_ax.T,base_d.T)
axs[0,1].plot(b1_ax.T,fa_d.T,'g')
axs[0,0].set_ylim([0,sim_ymax])
axs[0,1].set_ylim([0,experimental_ymax])
#axs[0,1].legend([r'FA=90$^\circ$',r'FA=45$^\circ$'],loc='center left',bbox_to_anchor=(1,0.5))
axs[0,1].legend([r'FA=90$^\circ$',r'FA=45$^\circ$'],loc='upper left')

# pyplot.savefig('figures_out/experimental.png',dpi=300)

pyplot.show()