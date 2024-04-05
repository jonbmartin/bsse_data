import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import sys
# sys.path.insert(1,'sigpy-rf-master/')
import sigpy.mri.rf as rf
from sigpy import backend
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.gridspec import GridSpec

font = {
        'size'   : 12}
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

matplotlib.rc('font', **font)


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


dt = 6e-6
pbc=1.4
pbw=0.35
bs_off=10000
fig = plt.figure(figsize=(16,5.5), dpi=80)
gs = GridSpec(1, 12, figure=fig)
# bot first (ax1-3) then top
ax1 = fig.add_subplot(gs[0:4])
ax2 = fig.add_subplot(gs[4:8], projection='3d')
ax3 = fig.add_subplot(gs[8:12], projection='3d')
# ax4 = fig.add_subplot(gs[12:16], projection='3d')


ax2.view_init(25, 250)
ax3.view_init(25, 250)
# ax4.view_init(25, 250)



rfp_bs, rfp_ss, _ = rf.dz_bssel_rf(dt=dt, tb=2, ndes=256, ptype='inv', flip=np.pi, pbw=pbw,
                                   pbc=[pbc], d1e=0.1, d2e=0.01,
                                   rampfilt=False, bs_offset=bs_off)



####### MOVING PLOTS
full_pulse = np.squeeze(rfp_bs )
T = dt * np.size(full_pulse)*1000
t = np.linspace(0,1, np.size(full_pulse))*T

b1_passband = 1
b1_stopband = 1
b1_multiphoton = 1

a, b = abrm_hp_collect(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse)))*b1_passband, np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1=None)

m_v_pass = get_3d_motion(a, b)
mx_pass = np.squeeze(m_v_pass[0, :])
my_pass = np.squeeze(m_v_pass[1, :])
mz_pass = np.squeeze(m_v_pass[2, :])

a, b = abrm_hp_collect(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse)))*b1_stopband, np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1=None)

m_v_stop = get_3d_motion(a, b)
mx_stop = np.squeeze(m_v_stop[0, :])
my_stop = np.squeeze(m_v_stop[1, :])
mz_stop = np.squeeze(m_v_stop[2, :])

a, b = abrm_hp_collect(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse)))*b1_multiphoton, np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1=None)

m_v_multiphoton = get_3d_motion(a, b)
mx_multiphoton = np.squeeze(m_v_multiphoton[0, :])
my_multiphoton = np.squeeze(m_v_multiphoton[1, :])
mz_multiphoton = np.squeeze(m_v_multiphoton[2, :])



ax1.set_title(u'A)  |$b_{bs}(t) + b_{ex}(t)$|')
ax1.set_ylabel('a.u.')
ax1.set_xlabel('t (ms)')
ax1.set_xlim(0, np.max(t))
ax1.set_ylim(0, 1.25)
plt.setp(ax1.get_xticklabels(),visible=True)

ax2.set_title(r'B)  $B_1^+$=0.17 mT (stopband)')
ax2.set_xlim3d(-1, 1)
ax2.set_ylim3d(-1, 1)
ax2.set_zlim3d(-1, 1)
ax2.set_xticks([-1, 0 ,1])
ax2.set_yticks([-1, 0 ,1])
ax2.set_zticks([-1, 0 ,1])
ax2.set_xlabel('$M_x$')
ax2.set_ylabel('$M_y$')
ax2.set_zlabel('$M_z$')


ax3.set_title(r'C)  $B_1^+$=0.14 mT (passband)')
ax3.set_xlim3d(-1, 1)
ax3.set_ylim3d(-1, 1)
ax3.set_zlim3d(-1, 1)
ax3.set_xticks([-1, 0 ,1])
ax3.set_yticks([-1, 0 ,1])
ax3.set_zticks([-1, 0 ,1])
ax3.set_xlabel('$M_x$')
ax3.set_ylabel('$M_y$')
ax3.set_zlabel('$M_z$')

# ax4.set_title(r'D)  $B_1^+$=4.14G (N=3 multiphoton)')
# ax4.set_xlim3d(-1, 1)
# ax4.set_ylim3d(-1, 1)
# ax4.set_zlim3d(-1, 1)
# ax4.set_xticks([-1, 0 ,1])
# ax4.set_yticks([-1, 0 ,1])
# ax4.set_zticks([-1, 0 ,1])
# ax4.set_xlabel('$M_x$')
# ax4.set_ylabel('$M_y$')
# ax4.set_zlabel('$M_z$')

lines = []
full_pulse = abs(full_pulse)
for i in range(len(t)):
    head = i - 1
    head_slice = (t > t[i] - 0.1) & (t < t[i]) # show last 10 us of pulse
    line1,  = ax1.plot(t[:i], full_pulse[:i], color='black')
    line1a, = ax1.plot(t[head_slice], full_pulse[head_slice], color='red', linewidth=2)
    line1e, = ax1.plot(t[head], full_pulse[head], color='red', marker='o', markeredgecolor='r')
    line2,  = ax2.plot3D(mx_stop[:i], my_stop[:i], mz_stop[:i], color='black')
    line2a, = ax2.plot3D(mx_stop[head_slice], my_stop[head_slice], mz_stop[head_slice], color='red', linewidth=2)
    line2e, = ax2.plot3D(mx_stop[head], my_stop[head], mz_stop[head], color='red', marker='o', markeredgecolor='r')
    line3,  = ax3.plot3D(mx_pass[:i], my_pass[:i], mz_pass[:i], color='black')
    line3a, = ax3.plot3D(mx_pass[head_slice], my_pass[head_slice], mz_pass[head_slice], color='red', linewidth=2)
    line3e, = ax3.plot3D(mx_pass[head], my_pass[head], mz_pass[head], color='red', marker='o', markeredgecolor='r')
    # line4,  = ax4.plot3D(mx_multiphoton[:i], my_multiphoton[:i], mz_multiphoton[:i], color='black')
    # line4a, = ax4.plot3D(mx_multiphoton[head_slice], my_multiphoton[head_slice], mz_multiphoton[head_slice], color='red', linewidth=2)
    # line4e, = ax4.plot3D(mx_multiphoton[head], my_multiphoton[head], mz_multiphoton[head], color='red', marker='o', markeredgecolor='r')
    # lines.append([line1,line1a,line1e,line2,line2a,line2e, line3, line3a,
    #               line3e, line4, line4a, line4e])
    #lines.append([line1,line1a,line1e,line2,line2a,line2e, line3, line3a,
                  #line3e])


    #+ rfp_ss





# Build the animation using ArtistAnimation function

ani = animation.ArtistAnimation(fig,lines,interval=20,blit=False)

plt.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.25)
plt.show()

print('Saving Figure')
fn = 'my_pulse_animation'
ani.save('%s.mp4'%(fn),writer='ffmpeg')
# ani.save('%s.gif'%(fn),writer='imagemagick')

# import subprocess
# cmd = 'magick convert %s.gif -fuzz 10%% -layers Optimize %s_r.gif'%(fn,fn)
# subprocess.check_output(cmd)
#
# plt.rcParams['animation.html'] = 'html5'