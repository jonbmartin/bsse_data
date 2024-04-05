import sigpy as sp
import numpy as np

import numpy as np
import sigpy as sp
from sigpy.mri.rf import slr as slr
from sigpy.mri.rf.util import dinf, b12wbs, calc_kbs, wbs2b1
from scipy.interpolate import interp1d
import sigpy.mri.rf as rf
import sigpy.plot as pl
import matplotlib.pyplot as pyplot

def dz_bssel_rf(dt=2e-6, tb=4, short_rat=1, ndes=128, ptype='ex', flip=np.pi/4,
                pbw=0.25, pbc=[1], d1e=0.01, d2e=0.01,
                rampfilt=True, bs_offset=20000,
                fa_correct=True,):
    """Design a math:`B_1^{+}`-selective pulse following J Martin's
    Bloch Siegert method.

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        short_rat (float): ratio of duration of desired pulse to duration
            required by nyquist. Can shorten pulse at expense of profile.
        ndes (int): number of taps in filter design.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians. Only required for ptype 'st',
            implied for other ptypes.
        pbw (float): width of passband in Gauss.
        pbc (list of floats): center of passband(s) in Gauss.
        d1e (float): passband ripple level in :math:`M_0^{-1}`.
        d2e (float): stopband ripple level in :math:`M_0^{-1}`.
        rampfilt (bool): option to directly design the modulated filter, to
            compensate b1 variation across a slice profile.
        bs_offset (float): (Hz) constant offset during pulse.
        fa_correct (bool): option to apply empirical flip angle correction.

    Returns:
        3-element tuple containing

        - **bsrf** (*array*): complex bloch-siegert gradient waveform.
        - **rfp** (*array*): complex slice-selecting waveform.
        - **rw** (*array*): complex bloch-siegert rewinder

    References:
        Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
        Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
        Proc. Intl. Soc. Magn. Reson. Med.
    """

    beta = 0.5  # AM waveform parameter for fermi sweeps # JBM was 0.5
    nsw = np.round(1250e-6 / dt)  # number of time points in sweeps
    kappa = np.arctan(2)  # FM waveform parameter

    # calculate bandwidth and pulse duration using lowest PBC of bands. Lower
    # PBC's require a longer pulse, so lowest constrains our pulse length
    upper_b1 = min(pbc) + pbw / 2
    lower_b1 = min(pbc) - pbw / 2

    # using Ramsey's BS shift equation pre- w_rf >> gam*b1 approximation
    B = b12wbs(bs_offset, upper_b1) - b12wbs(bs_offset,lower_b1)
    Tex = (tb / B) * short_rat  # seconds, the entire pulse duration

    # perform the design of the BS far off resonant pulse
    bsrf, rw, phi_bs = bssel_bs(Tex, dt, bs_offset)

    # design pulse for number of bands desired
    if len(pbc) == 1:
        rfp, phi_ex = bssel_ex_slr(Tex, dt, tb, ndes, ptype, flip, pbw, pbc[0],
                                   d1e, d2e, rampfilt, bs_offset, fa_correct)

    # repeat design for multiple bands of excitation
    else:
        rfp = np.zeros((1, np.int(np.ceil(Tex / dt / 2) * 2)), dtype=complex)
        for ii in range(0, len(pbc)):
            upper_b1 = pbc[ii] + pbw / 2
            lower_b1 = pbc[ii] - pbw / 2
            B_i = bs_offset * ((1 + (4258 * upper_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1) - \
                bs_offset * ((1 + (4258 * lower_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1)
            T_i = tb / B_i  # seconds, the entire pulse duration
            ex_subpulse = bssel_ex_slr(T_i, dt, tb, ndes, ptype, flip, pbw,
                                       pbc[ii], d1e, d2e, rampfilt,
                                       bs_offset)

            # zero pad to match the length of the longest pulse
            if ii > 0:
                zpad = np.zeros((1, np.size(rfp)-np.size(ex_subpulse)))
                zp1 = zpad[:, :np.size(zpad)//2]
                zp2 = zpad[:, (np.size(zpad))//2:]

                ex_subpulse = np.concatenate([zp1, ex_subpulse, zp2], axis=1)
            rfp += ex_subpulse

    # zero-pad it to the same length as bs
    nsw = int(np.ceil((np.size(bsrf) - np.size(rfp))/2))
    rfp = np.concatenate([np.zeros((1, np.int(nsw))), rfp], axis=1)
    rfp = np.concatenate([rfp,np.zeros((1,np.size(bsrf)-np.size(rfp)))], axis=1)

    # return the subpulses. User should superimpose bsrf and rfp if desired
    return bsrf, rfp, rw


def bssel_bs(T, dt, bs_offset):
    """Design the Bloch-Siegert shift inducing component pulse for a
     math:`B_1^{+}`-selective pulse following J Martin's Bloch Siegert method.

        Args:
            T (float): total pulse duration (s).
            dt (float): hardware sampling dwell time (s).
            bs_offset (float): constant offset during pulse (Hz).

        Returns:
            2-element tuple containing

            - **bsrf** (*array*): complex BS pulse.
            - **bsrf_rew** (*array*): FM waveform (radians/s).

        References:
            Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
            Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
            Proc. Intl. Soc. Magn. Reson. Med.
        """

    a = 0.00006
    Bth = 0.95
    t0 = T/2 - a*np.log((1-Bth)/Bth)
    T_full = 2*t0 +13.81 * a
    t = np.arange(-T_full/2, T_full/2, dt)
    bs_am = 1 / (1 + np.exp((np.abs(t)-t0)/a))
    if np.mod(np.size(bs_am), 2) != 0:
        bs_am = bs_am[:-1]

    A_half = bs_am[0:int(np.size(bs_am)/2)]
    gam = 4258
    k = 0.2
    om = (gam*A_half)/np.sqrt((1-(gam*A_half*dt)/k)**(-2)-1)
    alpha_t = np.arctan(np.sqrt((1-(gam*A_half*dt)/k)**(-2)-1)/gam)
    # import sigpy.plot as pl
    # pl.LinePlot(np.diff(alpha_t))
    # import matplotlib.pyplot as pyplot
    # pyplot.plot(om)
    # pyplot.plot(A_half*20500)
    # pyplot.show()
    # om = A_half*20500
    # import sigpy.plot as pl
    # pl.LinePlot(om)
    om -= np.max(abs(om))
    om = np.expand_dims(om*1,0)
    bs_fm = np.concatenate([-om, np.fliplr(-om)],axis=1) + bs_offset
    bs_fm_2 = bs_fm*2 # a 5 kHz increase for half of the RF pulse
    b1eff = np.sqrt(bs_fm**2 + (4258*bs_am)**2)/4258
    # # pyplot.plot(np.diff(alpha_t))
    # pyplot.plot(b1eff.T)
    # # pyplot.legend('dalpha/dt','b1eff')
    # pyplot.show()
    kbs_bs = calc_kbs(bs_am, bs_fm, T)

    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))
    bsrf2 = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm_2))
    bsrf2 = np.expand_dims(bsrf2,0)
    bsrf = np.expand_dims(bsrf,0)
    phi_bs = np.cumsum((4258*bs_am)**2/(2*bs_fm))

    # Build an RF rewinder, same amplitude but shorter duration to produce -0.5
    # the Kbs. Pull middle samples until duration matched
    bs_am_rew = np.ndarray.tolist(np.squeeze(bs_am))
    bs_fm_rew = np.ndarray.tolist(np.squeeze(-bs_fm))
    kbs_rw = -kbs_bs
    while abs(kbs_rw) > 0.5 * abs(kbs_bs):
        mid = len(bs_am_rew)//2
        bs_am_rew = bs_am_rew[:mid] + bs_am_rew[mid+1:]
        bs_fm_rew = bs_fm_rew[:mid] + bs_fm_rew[mid+1:]
        kbs_rw = calc_kbs(bs_am_rew, bs_fm_rew, len(bs_am_rew)*dt)

    # adjust amplitude to precisely give correct Kbs
    bs_am_rew = np.array(bs_am_rew) * np.sqrt(abs(kbs_bs/(2*kbs_rw)))
    kbs_rw = calc_kbs(bs_am_rew, bs_fm_rew, len(bs_am_rew) * dt)
    bsrf_rew = np.array(bs_am_rew) * np.exp(1j * dt * 2 * np.pi * np.cumsum(np.array(bs_fm_rew)))
    print('RW kbs = {}'.format(kbs_rw))

    pl.LinePlot(bsrf)
    pl.LinePlot(bsrf2)
    pl.LinePlot((bsrf+bsrf2))
    return (bsrf+bsrf2), bsrf_rew, phi_bs


def bssel_ex_slr(T, dt=2e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/2,
                 pbw=0.25, pbc=1, d1e=0.01, d2e=0.01, rampfilt=True,
                 bs_offset=20000, fa_correct=True):

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    if not rampfilt:
        # straightforward SLR design, no ramp
        rfp = slr.dzrf(ndes, tb, ptype, 'ls', d1e, d2e)
        rfp = np.expand_dims(rfp, 0)
    else:
        # perform a filtered design that compensates the b1 variation across
        # the slice. Here, calc parameter relations
        bsf, d1, d2 = slr.calc_ripples(ptype, d1e, d2e)

        # create a beta that corresponds to a ramp
        b = slr.dz_ramp_beta(ndes, T, ptype, pbc, pbw, bs_offset, tb, d1, d2, dt)

        if ptype == 'st':
            rfp = b
        else:
            # inverse SLR transform to get the pulse
            b = bsf * b
            rfp = slr.b2rf(np.squeeze(b))
            rfp = np.expand_dims(rfp, 0)

    # interpolate to target dwell time
    rfinterp = interp1d(np.linspace(-T / 2, T / 2, ndes), rfp, kind='cubic')
    trf = np.linspace(-T / 2, T / 2, n)
    rfp = rfinterp(trf)
    rfp = rfp * ndes / n

    # scale for desired flip if ptype 'st'
    if ptype == 'st':
        rfp = rfp / np.sum(rfp) * flip / (2 * np.pi * 4258 * dt)  # gauss
    else:  # rf is already in radians in other cases
        rfp = rfp / (2 * np.pi * 4258 * dt)

    # slice select modulation is middle of upper and lower b1
    upper_b1 = pbc + pbw / 2
    lower_b1 = pbc - pbw / 2
    rfp_modulation = 0.5*(b12wbs(bs_offset, upper_b1) + b12wbs(bs_offset, lower_b1))
    print(f'SS modulation = {rfp_modulation} Hz')

    # empirical correction factor for scaling
    if fa_correct:

        scalefact = pbc*(0.3323*np.exp(-0.9655*(rfp_modulation/bs_offset))
                         + 0.6821*np.exp(-0.02331*(rfp_modulation/bs_offset)))
        rfp = rfp / scalefact
    else:
        rfp = rfp / pbc

    # modulate RF to be centered at the passband. complex modulation => 1 band!
    t = np.linspace(- np.int(T / dt / 2), np.int(T / dt / 2), np.size(rfp))
    rfp = rfp * np.exp(-1j * 2 * np.pi * rfp_modulation * t * dt)

    phi_bs = np.cumsum((4258*np.real(rfp))**2/(2*rfp_modulation))
    return rfp, phi_bs


dt = 1e-6
db1 = 0.01
pbc = 1.5
b1min = 0
b1max = 5
b1 = np.arange(b1min, b1max, db1)  # gauss, b1 range to sim over

pbw = 0.5 # b1 (Gauss)
d1 = 0.01
d2 = 0.01
tb = 4
bs_offset = 5000

rfp_bs, rfp_ss, _ = dz_bssel_rf(dt=dt, tb=tb, ndes=256, ptype='st', flip=np.pi / 4, pbw=pbw,
                                pbc=[pbc], d1e=d1, d2e=d2,
                                rampfilt=True, bs_offset=bs_offset)
full_pulse = rfp_bs + rfp_ss
pl.LinePlot(rfp_bs)
print('Pulse duration = {}'.format(np.size(full_pulse)*dt))

a, b = rf.abrm_hp(2*np.pi*4258*dt*full_pulse.reshape((1, np.size(full_pulse))), np.zeros(np.size(full_pulse)),
                    np.array([[1]]), 0, b1.reshape(np.size(b1), 1))


Mxyfull = 2 * np.conj(a) * b

pyplot.figure()
pyplot.plot(b1, np.abs(Mxyfull.transpose()))
pyplot.ylabel('|Mxy|')
pyplot.xlabel('Gauss')
pyplot.ylim([0,1.05])
pyplot.xlim([b1min,b1max])
pyplot.show()