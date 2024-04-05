# -*- coding: utf-8 -*-
""":math:`B_1^{+}`-selective RF Pulse Design functions.

"""
import numpy as np
from sigpy.mri.rf import slr as slr
from sigpy.mri.rf import adiabatic as adiabatic
from sigpy.mri.rf.util import dinf
from scipy.interpolate import interp1d

__all__ = ['dz_bssel_rf', 'bssel_bs', 'bssel_bs_noflat', 'dz_bssel_chirp_rf', 'dz_b1_rf',
           'dz_b1_gslider_rf', 'dz_b1_hadamard_rf', 'calc_kbs']


def dz_bssel_rf(dt=2e-6, tb=4, short_rat=1, ndes=128, ptype='ex', flip=np.pi/4,
                pbw=0.25, pbc=[1], d1e=0.01, d2e=0.01,
                rampfilt=True, bs_offset=20000, rewinder=False):

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
        n_bands (int): number of bands if multibanding.
        band_sep (float): band separation if multibanding, in Gauss.
        d1e (float): passband ripple level in :math:`M_0^{-1}`.
        d2e (float): stopband ripple level in :math:`M_0^{-1}`.
        rampfilt (bool): option to directly design the modulated filter, to
            compensate b1 variation across a slice profile.
        bs_offset (float): (Hz) constant offset during pulse.
        rewinder (bool): option to add a Bloch-Siegert rewinder.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).



    References:
        Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
        Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
        Proc. Intl. Soc. Magn. Reson. Med.
    """

    beta = 4  # AM waveform parameter, you want this to reach ~1 @ end of pulse
    dw0 = 2 * np.pi * bs_offset  # amplitude of sweep
    nsw = np.round(500e-6 / dt)  # number of time points in sweeps
    kappa = np.arctan(4)  # FM waveform parameter

    # calculate bandwidth and pulse duration using lowest PBC of bands. Lower
    # PBC's require a longer pulse, so lowest constrains our pulse length
    upper_b1 = min(pbc) + pbw / 2
    lower_b1 = min(pbc) - pbw / 2

    # using Ramsey's BS shift equation pre- w_rf >> gam*b1 approximation
    B = bs_offset*((1+(4258*upper_b1)**2/bs_offset**2)**(1/2)-1) -\
        bs_offset*((1+(4258*lower_b1)**2/bs_offset**2)**(1/2)-1)
    short_rat=1
    T = (tb / B) * short_rat  # seconds, the entire pulse duration

    # perform the design of the BS far off resonant pulse
    bsrf, rw = bssel_bs(T, dt, nsw, dw0, bs_offset, beta, kappa)
    # bsrf_rw = bssel_adiab_bs(T, dt, nsw, dw0, bs_offset, beta, kappa)

    # design pulse for number of bands desired
    if len(pbc) == 1:
        rfp = bssel_ex_slr(T, dt, tb, ndes, ptype, flip, pbw, pbc[0], d1e, d2e,
                           rampfilt, bs_offset)
        rfp /= pbc[0]
    else:
        rfp = np.zeros((1, np.int(np.ceil(T / dt / 2) * 2)), dtype=complex)
        for ii in range(0, len(pbc)):
            upper_b1 = pbc[ii] + pbw / 2
            lower_b1 = pbc[ii] - pbw / 2
            B_i = bs_offset * ((1 + (4258 * upper_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1) - \
                bs_offset * ((1 + (4258 * lower_b1) ** 2 / bs_offset ** 2) ** (
                        1 / 2) - 1)
            T_i = tb / B_i  # seconds, the entire pulse duration
            ex_subpulse = bssel_ex_slr(T_i, dt, tb, ndes, ptype, flip, pbw,
                                       pbc[ii], d1e, d2e, rampfilt, bs_offset)

            # zero pad to match the length of the longest pulse
            if ii > 0:
                zpad = np.zeros((1, np.size(rfp)-np.size(ex_subpulse)))
                zp1 = zpad[:, :np.size(zpad)//2]
                zp2 = zpad[:, (np.size(zpad))//2:]

                ex_subpulse = np.concatenate([zp1, ex_subpulse, zp2], axis=1)
            rfp += ex_subpulse / pbc[ii]

    # zero-pad it to the same length as bs
    rfp = np.concatenate([np.zeros((1, np.int(nsw))), rfp,
                         np.zeros((1, np.int(nsw)))], axis=1)

    # return the subpulses. User should superimpose bsrf and rfp if desired
    return bsrf, rfp, rw


def bssel_bs(T, dt, nsw, dw0, bs_offset, beta=4, kappa=np.arctan(4)):

    # build the AHP pulse for the sweeps in and out
    t = range(np.int(nsw)) / nsw
    a = np.expand_dims(np.tanh(beta * t), 0)
    om = np.expand_dims(dw0 * np.tan(kappa * (t - 1)) / np.tan(kappa), 0)

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    # build the complete BS pulse
    bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)

    bs_fm = np.concatenate([-om, np.zeros((1, n)),
                            -np.fliplr(om)], axis=1) / 2 / np.pi + bs_offset

    # import sigpy.plot as pl
    # pl.LinePlot(bs_fm)

    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))

    # build an RF rewinder, half the duration of the BS pulse w/ opposite freq
    bs_am_rew = np.concatenate([a, np.ones((1, np.int(n / 2))), np.fliplr(a)],
                               axis=1)
    bs_fm_rew = -np.concatenate([-om, np.zeros((1, np.int(n / 2))),
                                 -np.fliplr(om)],
                                axis=1) / 2 / np.pi + bs_offset
    bsrf_rew = bs_am_rew * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm_rew))

    return bsrf, bsrf_rew


def bssel_adiab_bs(T, dt, nsw, dw0, bs_offset, beta=4, kappa=np.arctan(4)):

    n = np.int(T/dt)
    gamma = 2 * np.pi * 42.58
    k = 42.  # design param, affects max in-band excitation
    b1p = 1
    # build the complete BS pulse
    bs_am = np.ones((1,np.int(n)))
    t = np.arange(1, n//2 + 1) * T / n


    om = gamma * b1p / np.sqrt((1 - gamma * b1p / k * t) ** -2 - 1)
    om = np.concatenate((om, om[::-1]))


    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(om))

    # build an RF rewinder, half the duration of the BS pulse w/ opposite freq
    bs_am_rew = np.concatenate([a, np.ones((1, np.int(n / 2))), np.fliplr(a)],
                               axis=1)
    bs_fm_rew = -np.concatenate([-om, np.zeros((1, np.int(n / 2))),
                                 -np.fliplr(om)],
                                axis=1) / 2 / np.pi + bs_offset
    bsrf_rew = bs_am_rew * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm_rew))

    return bsrf, bsrf_rew


def bssel_bs_noflat(T, dt, nsw, dw0, bs_offset, beta=4, kappa=np.arctan(4)):

    # build the AHP pulse
    t = range(np.int(nsw)) / nsw
    a = np.expand_dims(np.tanh(beta * t), 0)
    om = np.expand_dims(dw0 * np.tan(kappa * (t - 1)) / np.tan(kappa), 0)

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    # build the complete BS pulse
    bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)

    bs_fm = np.concatenate([-om, np.zeros((1, n)),
                            -np.fliplr(om)], axis=1) / 2 / np.pi + bs_offset

    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))

    # build an RF rewinder, half the duration of the BS pulse w/ opposite freq
    bs_am_rew = np.concatenate([a, np.ones((1, np.int(n / 2))), np.fliplr(a)],
                               axis=1)
    bs_fm_rew = -np.concatenate([-om, np.zeros((1, np.int(n / 2))),
                                 -np.fliplr(om)],
                                axis=1) / 2 / np.pi + bs_offset
    bsrf_rew = bs_am_rew * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm_rew))

    return bsrf, bsrf_rew


def bssel_ex_slr(T, dt=2e-6, tb=4, ndes=128, ptype='ex', flip=np.pi/4,
                 pbw=0.25, pbc=1, d1e=0.01, d2e=0.01, rampfilt=True,
                 bs_offset=20000):

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    if not rampfilt:
        rfp = slr.dzrf(ndes, tb, ptype, 'ls', d1e, d2e)
        rfp = np.expand_dims(rfp,0)
    else:
        # perform a filtered design that compensates the b1 variation across
        # the slice
        if ptype == 'st':
            bsf = 1
            d1 = d1e
            d2 = d2e
            beta_ratio = (pbc + pbw / 2) / (pbc - pbw / 2)  # ratio btwn slice edges
        elif ptype == 'ex':
            bsf = np.sqrt(1/2)
            d1 = np.sqrt(d1e/2)
            d2 = d2e/np.sqrt(2)
            # ratio
            beta_ratio = np.sin(np.pi / 4 * (pbc + pbw / 2) / pbc) / np.sin(np.pi / 4 * (pbc - pbw / 2) / pbc)
        elif ptype == 'sat':
            bsf = np.sqrt(1 / 2)
            d1 = d1e / 2
            d2 = np.sqrt(d2e)
            beta_ratio = np.sin(np.pi / 4 * (pbc + pbw / 2) / pbc) / np.sin(np.pi / 4 * (pbc - pbw / 2) / pbc)
        elif ptype == 'se':
            raise ValueError('Warning: better to set rampfilt=False for 180 pulses')
        elif ptype == 'inv':
            raise ValueError('Warning: better to set rampfilt=False for 180 pulses')
        else:
            raise ValueError('Unknown pulse type. Recognized pulse types are st, ex, se, inv, and sat')

        # perform SLR design

        # create a beta that corresponds to a ramp
        b = slr.dz_ramp_beta(ndes, beta_ratio, tb, d1, d2)
        b = np.fliplr(b)

        if ptype == 'st':
            rfp = b
        else:
            # inverse SLR transform to get the pulse
            b = bsf * b
            rfp = slr.b2rf(np.squeeze(b))
            rfp = np.expand_dims(rfp,0)

    # interpolate to target dwell time
    rfinterp = interp1d(np.linspace(-T / 2, T / 2, ndes), rfp,
                   kind='cubic')
    trf = np.linspace(-T / 2, T / 2, n)
    rfp = rfinterp(trf)
    rfp = rfp * ndes / n

    # scale for desired flip if ptype 'st'
    if ptype == 'st':
        rfp = rfp / np.sum(rfp) * flip / (2 * np.pi * 4258 * dt)  # gauss
    else:  # rf is already in radians in other cases
        rfp = rfp / (2 * np.pi * 4258 * dt)

    # Again, using Ramsey's 1955 full eqn, not the approximation
    rfp_modulation = bs_offset*((1+(4258*pbc)**2/bs_offset**2)**(1/2)-1)

    # modulate RF to be centered at the passband. complex modulation => 1 band!
    rfp = rfp * np.exp(-1j * 2 * np.pi * rfp_modulation * np.linspace(- np.int(T / dt / 2), np.int(T / dt / 2), np.size(rfp)) * dt)

    return rfp


def dz_bssel_chirp_rf(dt=2e-6, T=0.005, pbb=0.25, pbt=1, bs_offset=20000):

    """Design a math:`B_1^{+}`-selective pulse following J Martin's unpublished
    Bloch Siegert method, using a chirp pulse excitation

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        ndes (int): number of taps in filter design.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians. Only required for ptype 'st',
            implied for other ptypes.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1e (float): passband ripple level in :math:`M_0^{-1}`.
        d2e (float): stopband ripple level in :math:`M_0^{-1}`.
        rampfilt (bool): option to directly design the modulated filter, to
            compensate b1 variation across a slice profile.
        bs_offset (float): (Hz) constant offset during pulse.
        rewinder (bool): option to add a Bloch-Siegert rewinder.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).



    References:
        Martin, J., Vaughn, C., Griswold, M., & Grissom, W. (2021).
        Bloch-Siegert |B 1+ |-Selective Excitation Pulses.
        Proc. Intl. Soc. Magn. Reson. Med.
    """

    beta = 4  # AM waveform parameter, you want this to reach ~1 @ end of pulse
    dw0 = 2 * np.pi * bs_offset  # amplitude of sweep
    nsw = np.round(500e-6 / dt)  # number of time points in sweeps
    kappa = np.arctan(4)  # FM waveform parameter

    # build the AHP pulse
    t = range(np.int(nsw))/nsw
    a = np.expand_dims(np.tanh(beta * t), 0)
    om = np.expand_dims(dw0 * np.tan(kappa * (t-1)) / np.tan(kappa), 0)

    # using Ramsey's BS shift equation pre- w_rf >> gam*b1 approximation
    B = bs_offset*((1+(4258*pbt)**2/bs_offset**2)**(1/2)-1) -\
        bs_offset*((1+(4258*pbb)**2/bs_offset**2)**(1/2)-1)

    n = np.int(np.ceil(T / dt / 2) * 2)  # samples in final pulse, force even

    # build the complete BS pulse
    bs_am = np.concatenate([a, np.ones((1, n)), np.fliplr(a)], axis=1)

    bs_fm = np.concatenate([-om, np.zeros((1, n)),
                            -np.fliplr(om)], axis=1)/2/np.pi+bs_offset
    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))

    # build an RF rewinder, half the duration of the BS pulse w/ opposite freq
    bs_am_rew = np.concatenate([a, np.ones((1, np.int(n/2))), np.fliplr(a)],
                               axis=1)
    bs_fm_rew = -np.concatenate([-om, np.zeros((1, np.int(n/2))),
                                 -np.fliplr(om)], axis=1)/2/np.pi+bs_offset
    bsrf_rew = bs_am_rew * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm_rew))


    # Again, using Ramsey's 1955 full eqn, not the approximation
    rfp_modulation_pbt = bs_offset*((1+(4258*pbt)**2/bs_offset**2)**(1/2)-1)
    rfp_modulation_pbb = bs_offset*((1+(4258*pbb)**2/bs_offset**2)**(1/2)-1)

    a, om = adiabatic.wurst(n, n_fac=40, bw=40000, dur=dt*n)
    t = np.arange(0, n) * dt
    n_fac = 5
    a = 1 - np.power(np.abs(np.cos(np.pi * t / (dt*n))), n_fac)
    om = np.linspace(rfp_modulation_pbb,rfp_modulation_pbt, n)

    # modulate RF to be centered at the passband. complex modulation => 1 band!
    rfp = a * np.exp(-1j * 2 * np.pi * np.cumsum(om) * dt)
    rfp = np.expand_dims(rfp, 0)
    # scale
    #rfp = rfp / np.sum(rfp) * flip / (2 * np.pi * 4258 * dt)  # gauss

    # zero-pad it to the same length as bs
    rfp = np.concatenate([np.zeros((1, np.int(nsw))), rfp,
                         np.zeros((1, np.int(nsw)))], axis=1)

    rfp = rfp / (pbt+pbb)/2/1.9
    # rfp = bsrf + rfp /(16*(pbt+pbb)/2)  # lastly, superimpose the pulses

    return bsrf, rfp


def dz_b1_rf(dt=2e-6, tb=4, ptype='st', flip=np.pi / 6, pbw=0.3,
             pbc=2, d1=0.01, d2=0.01, os=8, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective excitation pulse following Grissom \
    JMR 2014

    Args:
        dt (float): hardware sampling dwell time in s.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        os (int): matrix scaling factor.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    if pbc == 0:
        # we want passband as close to zero as possible.
        # do my own dual-band filter design to minimize interaction
        # between the left and right bands

        # build system matrix
        A = np.exp(1j * 2 * np.pi *
                   np.outer(np.arange(-n * os / 2, n * os / 2),
                            np.arange(-n / 2, n / 2)) / (n * os))

        # build target pattern
        ii = np.arange(-n * os / 2, n * os / 2) / (n * os) * 2
        w = dinf(d1, d2) / tb
        f = np.asarray([0, (1 - w) * (tb / 2),
                        (1 + w) * (tb / 2),
                        n / 2]) / (n / 2)
        d = np.double(np.abs(ii) < f[1])
        ds = np.double(np.abs(ii) > f[2])

        # shift the target pattern to minimum center position
        pbc = np.int(np.ceil((f[2] - f[1]) * n * os / 2 + f[1] * n * os / 2))
        dl = np.roll(d, pbc)
        dr = np.roll(d, -pbc)
        dsl = np.roll(ds, pbc)
        dsr = np.roll(ds, -pbc)

        # build error weight vector
        w = dl + dr + d1 / d2 * np.multiply(dsl, dsr)

        # solve for the dual-band filter
        AtA = A.conj().T @ np.multiply(np.reshape(w, (np.size(w), 1)), A)
        Atd = A.conj().T @ np.multiply(w, dr - dl)
        h = np.imag(np.linalg.pinv(AtA) @ Atd)

    else:  # normal design

        # design filter
        h = slr.dzls(n, tb, d1, d2)

        # dual-band-modulate the filter
        om = 2 * np.pi * 4257 * pbc  # modulation frequency
        t = np.arange(0, n) * pulse_len / n - pulse_len / 2
        h = 2 * h * np.sin(om * t)

    if split_and_reflect:
        # split and flip fm waveform to improve large-tip accuracy
        dom = np.concatenate((h[n // 2::-1], h, h[n:n // 2:-1])) / 2
    else:
        dom = np.concatenate((0 * h[n // 2::-1], h, 0 * h[n:n // 2:-1]))

    # scale to target flip, convert to Hz
    dom = dom * flip / (2 * np.pi * dt)

    # build am waveform
    om1 = np.concatenate((-np.ones(n // 2), np.ones(n), -np.ones(n // 2)))

    return om1, dom


def dz_b1_gslider_rf(dt=2e-6, g=5, tb=12, ptype='st', flip=np.pi / 6,
                     pbw=0.5, pbc=2, d1=0.01, d2=0.01, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective excitation gSlider pulse following
     Grissom JMR 2014.

    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
     tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    # if ptype == 'st':
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    om = 2 * np.pi * 4257 * pbc  # modulation freq to center profile at pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf*slr.dz_gslider_b(n, g, gind, tb, d1, d2, np.pi, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = np.concatenate((h[n // 2::-1],
                                               h, h[n:n // 2:-1])) / 2
        else:
            dom[:, gind - 1] = np.concatenate((0 * h[n // 2::-1],
                                              h, 0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                          -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom


def dz_b1_hadamard_rf(dt=2e-6, g=8, tb=16, ptype='st', flip=np.pi / 6,
                      pbw=2, pbc=2, d1=0.01, d2=0.01, split_and_reflect=True):
    """Design a :math:`B_1^{+}`-selective Hadamard-encoded pulse following \
     Grissom JMR 2014.
    Args:
        dt (float): hardware sampling dwell time in s.
        g (int): number of slabs to be acquired.
        tb (int): time-bandwidth product.
        ptype (string): pulse type, 'st' (small-tip excitation), 'ex' (pi/2 \
            excitation pulse), 'se' (spin-echo pulse), 'inv' (inversion), or \
            'sat' (pi/2 saturation pulse).
        flip (float): flip angle, in radians.
        pbw (float): width of passband in Gauss.
        pbc (float): center of passband in Gauss.
        d1 (float): passband ripple level in :math:`M_0^{-1}`.
        d2 (float): stopband ripple level in :math:`M_0^{-1}`.
        split_and_reflect (bool): option to split and reflect designed pulse.

    Split-and-reflect preserves pulse selectivity when scaled to excite large
    tip-angles.

    Returns:
        2-element tuple containing

        - **om1** (*array*): AM waveform.
        - **dom** (*array*): FM waveform (radians/s).

    References:
        Grissom, W., Cao, Z., & Does, M. (2014).
        :math:`B_1^{+}`-selective excitation pulse design using the Shinnar-Le
        Roux algorithm. Journal of Magnetic Resonance, 242, 189-196.
    """

    # calculate beta filter ripple
    [_, d1, d2] = slr.calc_ripples(ptype, d1, d2)
    bsf = flip

    # calculate pulse duration
    b = 4257 * pbw
    pulse_len = tb / b

    # calculate number of samples in pulse
    n = np.int(np.ceil(pulse_len / dt / 2) * 2)

    # modulation frequency to center profile at pbc gauss
    om = 2 * np.pi * 4257 * pbc
    t = np.arange(0, n) * pulse_len / n - pulse_len / 2

    om1 = np.zeros((2 * n, g))
    dom = np.zeros((2 * n, g))
    for gind in range(1, g + 1):
        # design filter
        h = bsf*slr.dz_hadamard_b(n, g, gind, tb, d1, d2, n // 4)

        # modulate filter to center and add it to a time-reversed and modulated
        # copy, then take the imaginary part to get an odd filter
        h = np.imag(h * np.exp(1j * om * t) - h[n::-1] * np.exp(1j * -om * t))
        if split_and_reflect:
            # split and flip fm waveform to improve large-tip accuracy
            dom[:, gind - 1] = np.concatenate((h[n // 2::-1],
                                              h,
                                              h[n:n // 2:-1])) / 2
        else:
            dom[:, gind - 1] = np.concatenate((0 * h[n // 2::-1],
                                              h,
                                              0 * h[n:n // 2:-1]))
        # build am waveform
        om1[:, gind - 1] = np.concatenate((-np.ones(n // 2), np.ones(n),
                                          -np.ones(n // 2)))

    # scale to target flip, convert to Hz
    dom = dom / (2 * np.pi * dt)

    return om1, dom


def calc_kbs(b1, wrf, T):
    """Calculate Kbs for a given pulse shape. Kbs is a constant that describes
    the phase shift (radians/Gauss^2) for a given RF pulse.
    Args:
        b1 (array): RF amplitude modulation, normalized.
        wrf (array): frequency modulation (Hz).
        T (float): pulse length (s)

    Returns:
        kbs (float): kbs constant for the input pulse, rad/gauss**2/msec

    References:
        Sacolick, L; Wiesinger, F; Hancu, I.; Vogel, M. (2010).
        B1 Mapping by Bloch-Siegert Shift. Magn. Reson. Med., 63(5): 1315-1322.
    """
    # squeeze just to ensure 1D
    b1 = np.squeeze(b1)
    wrf = np.squeeze(wrf)

    gam = 42.5657*2*np.pi*10**6  # rad/T
    t = np.linspace(0, T, np.size(b1))

    kbs = np.trapz(((gam*b1)**2/((2*np.pi*wrf)*2)),t)
    kbs /= (10000*10000)  # want out rad/G**2

    return kbs
