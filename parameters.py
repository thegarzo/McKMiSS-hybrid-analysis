
import numpy as np
# Here we will put cuts for different sets of observables. In general, these will be nice dictionaries with the relevant info, 

# ── global bin definitions (must match across all nodes) ───────────────────────

# pT bins: logarithmically spaced from 0.05 to 3.0 GeV.
# Log spacing gives finer resolution at low pT and coarser bins at high pT 
# We start at 0.05 GeV since particles below ~50 MeV/c are below the iSS sampling threshold anyway.
PT_MIN   = 0.05                            # lower edge [GeV]
PT_MAX   = 5.0                               # upper edge [GeV]
N_PT     = 60                                # number of bins
PT_BINS  = np.logspace(np.log10(PT_MIN),
                       np.log10(PT_MAX),
                       N_PT + 1)             # shape (61,) — 60 bin edges
PT_CENTS = np.sqrt(PT_BINS[:-1] * PT_BINS[1:])  # geometric centre of each bin

# ── coarse pT grid: for flow vn(pT) ───────────────────────────────────
# Fewer bins means more particles per bin, more stable Q-vectors.
# Typical ALICE flow analysis uses ~10-15 bins up to 3 GeV.
PT_MIN_FLOW  = 0.2
PT_MAX_FLOW  = 3.0
N_PT_FLOW    = 12
PT_BINS_FLOW = np.logspace(np.log10(PT_MIN_FLOW), np.log10(PT_MAX_FLOW),
                           N_PT_FLOW + 1)
PT_CENTS_FLOW = np.sqrt(PT_BINS_FLOW[:-1] * PT_BINS_FLOW[1:])

ALICE_TRACK_SEL = [PT_MIN,50.]

# ── eta bins for dNch/deta vs eta ─────────────────────────────────────
ETA_BINS  = np.linspace(-5.0, 5.0, 51)   # 120 bins, 0.5 wide
ETA_CENTS = 0.5 * (ETA_BINS[:-1] + ETA_BINS[1:])
N_ETA     = len(ETA_CENTS)

# Flow harmonics to compute.
# n=2: elliptic flow 
# n=3: triangular flow
# n=4: quadrangular flow
ORDERS   = [2, 3, 4]

# Particle species to track: name -> PDG Monte Carlo ID.
SPECIES = {
    # 'photon':     22,
    'pi_0':    111,
    'pi_plus':    211,
    'pi_minus':    -211,
    'kaon_plus':  321,
    'kaon_minus':  -321,
    'proton':     2212,
    'anti-proton':     -2212,
    'neutron':     2112,
    'anti-neutron': -2112,
    'charged_hadrons': [211, -211, 321, -321, 2212, -2212], # all charged hadrons
}

# Fancy Baryons
BARYONS={
    'Lambda': 3122,
    'Anti_Lambda': -3122,
    'Sigma_plus': 3222,
    'Sigma_zero': 3212,
    'Sigma_minus': 3112,
    'anti-Sigma_plus': -3222,
    'anti-Sigma_zero': -3212,
    'anti-Sigma_minus': -3112,
    'Xi_zero': 3322,
    'Xi_minus': 3312,
    'anti-Xi_zero': -3322,
    'anti-Xi_minus': -3312,
    'Omega': 3334,
    'anti-Omega': -3334,
}

# Rapidity window for all observables.
# |eta| < 1.0 covers midrapidity and is standard for LHC analyses.
# But we need to also consider other rapidity windows for analysis.
RAP_CUTS  ={
    0: [-0.8,0.8], # ALICE midrapidity
    1: [-3.3,-2.1], 
    2: [2.8,5.1],   # VZERO-A
    3: [-3.7,-1.7], # VZERO-C
    4: [-0.8,-0.4], # ALICE Flow Sub-event A
    5: [0.4,0.8], # ALICE Flow Sub-event B
} 


RAP_CUTS_ASSIGMENTS  ={
    "ALICE midrapidity": 0,
    "ALICE backward": 1,
    "VZERO-A": 2,
    "VZERO-C": 3,
    "ALICE Flow Sub-event A": 4,
    "ALICE Flow Sub-event B": 5,
} 