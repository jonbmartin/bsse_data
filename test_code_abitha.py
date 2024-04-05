import numpy as np
import matplotlib as plt

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
    t_v = np.arange(dt, T_full/2*dt+dt, dt)
    om = (gam*A_half)/np.sqrt((1-(gam*A_half*abs(t_v))/k)**(-2)-1)
    om -= np.max(abs(om))
    om = np.expand_dims(om*1,0)
    bs_fm = np.concatenate([-om, np.fliplr(-om)],axis=1) + bs_offset
    kbs_bs = calc_kbs(bs_am, bs_fm, T)
    bsrf = bs_am * np.exp(1j * dt * 2 * np.pi * np.cumsum(bs_fm))
    bsrf = np.expand_dims(bsrf,0)
    phi_bs = np.cumsum((4258*bs_am)**2/(2*bs_fm))
    print('kbs = {}'.format(kbs_bs))
    return bsrf, phi_bs

T = 8e-3;
dt = 1e-6;
Bsoffset = 4e3;
bsrf = bssel_bs(T, dt, Bsoffset)

am = np.ones((8000,1))
fm = np.ones((8000,1)) * 4000  #  Hz
dt = 1e-6

kbs = calc_kbs(am,fm,0.008)
print(kbs)
