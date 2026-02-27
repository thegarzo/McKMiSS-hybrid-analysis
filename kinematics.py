import numpy as np
# Here we will put cuts for different sets of observables. In general, these will be nice dictionaries with the relevant info, 

# ── global bin definitions (must match across all nodes) ───────────────────────

def get_kinematics(ev):
    px, py, pz, E = ev['px'], ev['py'], ev['pz'], ev['E']

    pt  = np.sqrt(px**2 + py**2)
    pv  = np.sqrt(px**2 + py**2 + pz**2)
    phi = np.arctan2(py, px)
    # Protect against log(0) or log(negative) for massless/near-massless
    # particles going exactly along the beam axis.
    # Replace (E - pz) <= 0 with infinity so the ratio -> 0 and rap -> -inf,
    # then NaN-mask those entries.
    # safe = np.where(E - pz > 0, (E + pz) / (E - pz), np.inf)
    # rap  = np.where(safe > 0, 0.5 * np.log(safe), np.nan)
    
    safe = np.where(pv - pz > 0, (pv + pz) / (pv - pz), np.inf)
    psrap  = np.where(safe > 0, 0.5 * np.log(safe), np.nan)

    return pt, phi, psrap


def select_species(ev, pdg_spec, rap_cut):
    """
    Parameters
    ----------
    ev      : structured array — one sampling event from read_iSS_binary()
    pdg     : int              — PDG Monte Carlo ID of desired species
    rap_cut : list            — keep particles with  rap_cut[0] < |y| < rap_cut[1]

    Returns
    -------
    pt, phi, rap : np.ndarray — kinematics of selected particles
    """
    if isinstance(pdg_spec, int):
        pdg_mask = ev['pid'] == pdg_spec
    else:
        pdg_mask = np.zeros(len(ev), dtype=bool)
        for pdg in pdg_spec:
            pdg_mask |= (ev['pid'] == pdg)

    pt, phi, rap = get_kinematics(ev)
    mask = pdg_mask & (rap > rap_cut[0]) & (rap < rap_cut[1]) 
    return pt[mask], phi[mask], rap[mask]
