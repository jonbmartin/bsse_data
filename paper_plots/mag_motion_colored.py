import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
from sigpy import backend

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation



def abrm_hp_collect(rf, gamgdt, xx, dom0dt=0, b1=None):
    device = backend.get_device(rf)
    xp = device.xp

    if b1 is None:
        rf = rf.flatten()

    with device:
        Ns = xx.shape[0] # Ns: # of spatial locs
        Nt = gamgdt.shape[0]  # Nt: # time points
        a_v = []
        b_v = []

        a = xp.ones((Ns,))
        b = xp.zeros((Ns,))

        for ii in xp.arange(Nt):
            # apply phase accural
            z = xp.exp(-1j * (xx * gamgdt[ii, ] + dom0dt))
            b = b * z

            # apply rf
            if b1 is None:
                C = xp.cos(xp.abs(rf[ii]) / 2)
                S = 1j * xp.exp(1j * xp.angle(rf[ii])) * xp.sin(xp.abs(rf[ii]) / 2)
            else:
                b1rf = b1 @ rf[:, ii]
                C = xp.cos(xp.abs(b1rf) / 2)
                S = 1j * xp.exp(1j * xp.angle(b1rf)) * xp.sin(xp.abs(b1rf) / 2)
            at = a * C - b * xp.conj(S)
            bt = a * S + b * C
            a_v.append(at)
            b_v.append(bt)

            a = at
            b = bt

        z = xp.exp(1j / 2 * (xx * xp.sum(gamgdt, axis=0) + Nt * dom0dt))
        a = a * z
        b = b * z

        return a_v, b_v

# design the pulse
dt = 2e-6
b1 = np.arange(0,10, 0.01)  # gauss, b1 range to sim over

pbc = 1.4 # b1 (Gauss)
pbw = 0.30 # b1 (Gauss)
bs_off = 7500

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='inv', flip=np.pi, pbw=pbw,
                                   pbc=[pbc], d1e=0.1, d2e=0.01,
                                   rampfilt=False, bs_offset=bs_off)
t = np.arange(0,dt*np.size(rfp_bs),dt) * 1000 # ms
nt = np.size(rfp_bs)
nsweep = 315  # approximate, just a guess really

full_pulse = np.squeeze(rfp_bs + rfp_ss)
pyplot.figure()
pyplot.plot(t[0:nsweep],abs(full_pulse[0:nsweep]),color='red')
pyplot.plot(t[nsweep:nt-nsweep],abs(full_pulse[nsweep:nt-nsweep]),color='blue')
pyplot.plot(t[nt-nsweep:],abs(full_pulse[nt-nsweep:]),color='red')
pyplot.xlim([-0.1, 3.75])
pyplot.ylim([-0.1,1.15])
pyplot.title(r'$|b_{bs}(t) + b_{ex}(t)|$')
# pyplot.title(r'$B_1^+$ = 0.17 mT (stopband)')

pyplot.xlabel('t (ms)')
pyplot.ylabel('(a.u.)')
# pyplot.title('BSSE Inversion Pulse')
pyplot.show()


print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b
pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.ylim([0,1])
pyplot.show()

mx_v = []
my_v = []
mz_v = []

# now sim at just one b1
b1_inspect = 1.7
# for ii in range(0,np.size(full_pulse),10):
#     rf_t = full_pulse[:,0:(ii)]
#     print(ii)
a, b = abrm_hp_collect(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse)))*b1_inspect, np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1=None)
a = np.array(a)
b = np.array(b)
# mx_v.append(np.squeeze(2*np.real(a*np.conj(b))))
# my_v.append(np.squeeze(2*np.imag(a*np.conj(b))))
# mz_v.append(np.squeeze(1-2*abs(b)**2))
mx_v = np.squeeze(2*np.real(a*np.conj(b)))
my_v = np.squeeze(2*np.imag(a*np.conj(b)))
mz_v = np.squeeze(1-2*abs(np.array(b))**2)

t = np.arange(0,dt*np.size(full_pulse),dt)  #s
w = 2*np.pi*bs_off
mx_v_r = mx_v * np.cos(w*t) - my_v * np.sin(w*t)
my_v_r = mx_v * np.sin(w*t) + my_v * np.cos(w*t)



fig = pyplot.figure()
ax = p3.Axes3D(fig)
mx_v = np.expand_dims(np.array(mx_v_r),0)
my_v = np.expand_dims(np.array(my_v_r),0)
mz_v = np.expand_dims(np.array(mz_v),0)
m_v = np.concatenate([mx_v, my_v, mz_v],0)
data = []
data.append(m_v[:,0:nsweep])
data.append(m_v[:,nsweep:nt-nsweep])
data.append(m_v[:,nt-nsweep:])


# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-1, 1])
ax.set_xlabel(r'$M_X$')
ax.set_xticks([-1, 0, 1])

ax.set_ylim3d([-1, 1])
ax.set_ylabel(r'$M_Y$')
ax.set_yticks([-1, 0, 1])

ax.set_zlim3d([-1, 1.0])
ax.set_zlabel(r'$M_Z$')
ax.view_init(20, 170)
ax.set_zticks([-1, 0, 1])




def update_lines_colored(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    lines[0].set_color('red')
    lines[1].set_color('blue')
    lines[2].set_color('red')

    # if i == magic_value: fig.savefig(f'fig{i}.png')

    return lines



# Creating the Animation object
ani = animation.FuncAnimation(fig, update_lines_colored, np.size(full_pulse), fargs=(data, lines),
                              interval=1, blit=True, repeat=False)

# ax = pyplot.axes()
# ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

# pyplot.draw()
pyplot.show()



# ani.save("rfmotion_nonresonant.png",writer="imagemagick")
# writergif = animation.PillowWriter(fps=80)
# line_ani.save('off-res.gif', writer=writergif)

# create a figure with two subplots
