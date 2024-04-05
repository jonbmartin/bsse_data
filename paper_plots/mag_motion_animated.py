import sigpy.mri.rf as rf
import numpy as np
import matplotlib.pyplot as pyplot
import sigpy.plot as pl
from sigpy import backend

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update_lines_animated(num, dataLines, lines) :
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines

def update_lines_colored(num, dataLines, lines) :
    #first sweep
    numsweep = 300
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :numsweep])
        line.set_3d_properties(data[2,:numsweep])
        line.set_color('red')
    #flat
    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, numsweep:2*numsweep])
        line.set_3d_properties(data[2,numsweep:2*numsweep])
        line.set_color('blue')
    return lines

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

def get_3d_motion(a, b):
    a = np.array(a)
    b = np.array(b)
    # mx_v.append(np.squeeze(2*np.real(a*np.conj(b))))
    # my_v.append(np.squeeze(2*np.imag(a*np.conj(b))))
    # mz_v.append(np.squeeze(1-2*abs(b)**2))
    mx_v = np.squeeze(2 * np.real(a * np.conj(b)))
    my_v = np.squeeze(2 * np.imag(a * np.conj(b)))
    mz_v = np.squeeze(1 - 2 * abs(np.array(b)) ** 2)

    t = np.arange(0, dt * np.size(full_pulse), dt)  # s
    w = 2 * np.pi * bs_off
    mx_v_r = mx_v * np.cos(w * t) - my_v * np.sin(w * t)
    my_v_r = mx_v * np.sin(w * t) + my_v * np.cos(w * t)

    mx_v = np.expand_dims(np.array(mx_v_r), 0)
    my_v = np.expand_dims(np.array(my_v_r), 0)
    mz_v = np.expand_dims(np.array(mz_v), 0)
    m_v = np.concatenate([mx_v, my_v, mz_v], 0)
    return m_v

dt = 2e-6
b1 = np.arange(0,10, 0.01)  # gauss, b1 range to sim over

pbc = 2 # b1 (Gauss)
pbw = 0.35 # b1 (Gauss)
bs_off = 7500

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='inv', flip=np.pi, pbw=pbw,
                                   pbc=[pbc], d1e=0.1, d2e=0.01,
                                   rampfilt=False, bs_offset=bs_off)

full_pulse = rfp_bs + rfp_ss
pl.LinePlot(full_pulse)

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
b1_inspect = pbc
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
data.append(m_v)

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]


# Setting the axes properties
ax.set_xlim3d([-1, 1])
ax.set_xlabel('X')

ax.set_ylim3d([-1, 1])
ax.set_ylabel('Y')

ax.set_zlim3d([-1, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')
ax.view_init(25, 200)

breaking = 100


# Creating the Animation object
ani = []

ani.append(animation.FuncAnimation(fig, update_lines_animated, np.size(full_pulse), fargs=(data, lines),
                              interval=0.05, blit=True))


pyplot.show()
# writergif = animation.PillowWriter(fps=80)
# line_ani.save('off-res.gif', writer=writergif)
