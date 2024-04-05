import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sigpy.mri.rf as rf
from sigpy import backend
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.gridspec import GridSpec

plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
font = {
        'size'   : 16}
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

matplotlib.rc('font', **font)


def get_arrow(dx, dy, dz):
    x = 0
    y = 0
    z = 0
    u = dx
    v = dy
    w = dz
    return x,y,z,u,v,w

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


def get_3d_motion(a, b, bs_off):
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

# design RF pulse
dt = 6e-6
pbc=1.4
pbw=0.35
bs_off=7500

rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='inv', flip=np.pi, pbw=pbw,
                                   pbc=[pbc], d1e=0.1, d2e=0.01,
                                   rampfilt=False, bs_offset=bs_off)
# full_pulse = np.squeeze(rfp_bs + rfp_ss)
full_pulse = np.squeeze(rfp_bs)
nt = np.size(full_pulse)
T = dt * nt * 1000
t = np.linspace(0,1, nt)*T

b1_passband = 1.4
b1_stopband = 1.7

# simulate pulse
a, b = abrm_hp_collect(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse)))*b1_passband, np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1=None)

m_v_pass = get_3d_motion(a, b, bs_off)
mx_pass = np.squeeze(m_v_pass[0, :])
my_pass = np.squeeze(m_v_pass[1, :])
mz_pass = np.squeeze(m_v_pass[2, :])



# plotting code
mx0, my0, mz0 = 0, 0, 1
quiver = ax.quiver(*get_arrow(mx0, my0, mz0), color='r')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

def update(t):
    global quiver
    quiver.remove()
    t = int(t)
    dx = mx_pass[t]
    dy = my_pass[t]
    dz = mz_pass[t]
    quiver = ax.quiver(*get_arrow(dx, dy, dz), color='r')

# plot some axes
axis_line = np.linspace(-1.25,1.25,100)
ax.plot3D(0*axis_line, 0*axis_line, axis_line, color='w',lw=2)
ax.plot3D(0*axis_line, 1*axis_line, 0*axis_line, color='w',lw=2)
ax.plot3D(1*axis_line, 0*axis_line, 0*axis_line, color='w',lw=2)

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.set_xticks([-1, 0 ,1])
ax.set_yticks([-1, 0 ,1])
ax.set_zticks([-1, 0 ,1])
ax.set_xlabel('$M_x$',fontsize='medium')
ax.set_ylabel('$M_y$', fontsize='medium')
ax.set_zlabel('$M_z$', fontsize='medium')

# JBM controls perspective on scene
ax.view_init(25, 145)

plt.title('*insert title here*', y=1)
ani = FuncAnimation(fig, update, frames=np.linspace(0,nt-1,nt-1), interval=50)
plt.tight_layout(pad=0., w_pad=0.25, h_pad=0.25)
# show the plot...can just close and will directly start saving same animation
plt.show()

print('Saving Figure')
fn = 'arrow_animation'
ani.save('%s.mp4'%(fn),writer='ffmpeg')
print('Done Saving')