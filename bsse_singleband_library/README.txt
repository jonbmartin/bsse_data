.mat file variables:

b1_sim: b1 grid over which pulse simulation is performed (G)
bsse_am: TNMR-ready AM table. BS component is normalized to 1
bsse_phase: TNMR-ready phase table. Phase values are degrees, [0 360]
dt: dwell time (s)
dur: pulse duration (s)
mxy: complex simulated profile excited by the pulse. Corresponds to 'b1sim'
pbc: passband center (G)
pbw: passband width (G)
raw_bs: full complex BS waveform
raw_ss: full complex SS waveform

General notes:
Bands are PBW = 1/8G, arranged from PBC=1G to PBC=3G
Longest pulse duration is PBC = 1G, 8.6 ms
Shortest pulse duration is PBC = 3G, 5.4 ms
All pulses are TB=2, 90dg flip