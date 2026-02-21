import numpy as np
import h5py
import sys

from parameters import *
import kinematics as kn



# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Q-vector summary computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(events):
    """
    Reduce all sampling events to per-pT-bin Q-vector summaries.

    For each sampling event and each species we compute:

        N_pt    [n_pt]         : particle count per pT bin
        sum_pt  [n_pt]         : sum of pT values per bin (-> <pT> when divided by N_pt)
        Qn_real [n_pt, n_ord]  : Re( sum_j e^{i n phi_j} ) for particles in bin
        Qn_imag [n_pt, n_ord]  : Im( ... )
        Q2n_real[n_pt, n_ord]  : Re( sum_j e^{2i n phi_j} ) — needed for vn{4}
        Q2n_imag[n_pt, n_ord]  : Im( ... )
        pQn_real[n_pt, n_ord]  : Re( sum_j pT_j e^{i n phi_j} ) — pT-weighted Q-vector
                                  needed for <vn * <pT>> correlator
        pQn_imag[n_pt, n_ord]  : Im( ... )

    From these 8 arrays per species you can reconstruct ALL desired observables
    during the final merged analysis without ever needing the particle list again.

    Why pT bins?
    Storing Qn per pT bin gives access to pT-differential flow vn(pT) via the
    scalar product method. The integrated Q-vectors are recovered for free by
    summing over all pT bins.

    Parameters
    ----------
    events : list of np.ndarray
        Output of read_iSS_binary().

    Returns
    -------
    dict : {species_name: {array_name: np.ndarray}}
        Outer key: species name string (e.g. 'pi_plus').
        Inner keys: 'N_pt', 'sum_pt', 'Qn_real', 'Qn_imag',
                    'Q2n_real', 'Q2n_imag', 'pQn_real', 'pQn_imag'.
        All arrays have leading dimension n_events (number of sampling events).
    """
    n_ev  = len(events)
    n_ord = len(ORDERS)
    out   = {}
    n_rap_cuts = len(RAP_CUTS)
    
    for name, pdg in SPECIES.items():

        # fine grid — for spectra
        N_pt      = np.zeros((n_ev, n_rap_cuts, N_PT),             dtype=np.int32)
        sum_pt    = np.zeros((n_ev, n_rap_cuts, N_PT),             dtype=np.float32)

        # coarse grid — for flow Q-vectors
        Qn_real   = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)
        Qn_imag   = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)
        Q2n_real  = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)
        Q2n_imag  = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)
        pQn_real  = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)
        pQn_imag  = np.zeros((n_ev, n_rap_cuts, N_PT_FLOW, n_ord), dtype=np.float32)

        # eta histogram — no rapidity window cut, covers full eta range
        # shape (n_ev, N_ETA): count all particles regardless of pT or rapidity window
        N_eta = np.zeros((n_ev, N_ETA), dtype=np.int32)

        for iev, ev in enumerate(events):
            
            # ── eta histogram: all particles of this species ───────────
            # use pseudorapidity approximation eta ~ y for massless particles,
            # or compute from px,py,pz directly if available.
            # Here we use rapidity y stored in select_species with wide window.
            # For a proper eta you would need pz and |p| from the event struct.
            px  = ev['px'][ev['pid'] == pdg] if isinstance(pdg, int) \
                  else ev['px'][np.isin(ev['pid'], pdg)]
            py  = ev['py'][ev['pid'] == pdg] if isinstance(pdg, int) \
                  else ev['py'][np.isin(ev['pid'], pdg)]
            pz  = ev['pz'][ev['pid'] == pdg] if isinstance(pdg, int) \
                  else ev['pz'][np.isin(ev['pid'], pdg)]

            if len(pz) > 0:
                # pseudorapidity: eta = -ln(tan(theta/2)) = 0.5*ln((|p|+pz)/(|p|-pz))
                p_abs  = np.sqrt(px**2 + py**2 + pz**2)
                safe   = np.where(p_abs - pz > 0, p_abs - pz, np.inf)
                eta    = np.where(safe < np.inf,
                                  0.5 * np.log((p_abs + pz) / safe),
                                  np.nan)
                in_range = (eta >= ETA_BINS[0]) & (eta < ETA_BINS[-1]) & np.isfinite(eta)
                if np.any(in_range):
                    counts, _ = np.histogram(eta[in_range], bins=ETA_BINS)
                    N_eta[iev] += counts.astype(np.int32)

            for RCUT in RAP_CUTS:
                RAP_CUT = RAP_CUTS[RCUT]
                pt, phi, _ = kn.select_species(ev, pdg, RAP_CUT)

                if len(pt) == 0:
                    continue

                # ── fine grid: spectra ─────────────────────────────────
                in_range_fine = (pt >= PT_BINS[0]) & (pt < PT_BINS[-1])
                pt_fine = pt[in_range_fine]

                if len(pt_fine) > 0:
                    ibin_fine = np.searchsorted(PT_BINS[1:], pt_fine)
                    for ib in range(N_PT):
                        sel = ibin_fine == ib
                        if not np.any(sel):
                            continue
                        N_pt  [iev, RCUT, ib] = sel.sum()
                        sum_pt[iev, RCUT, ib] = pt_fine[sel].sum()

                # ── coarse grid: flow Q-vectors ────────────────────────
                in_range_flow = (pt >= PT_BINS_FLOW[0]) & (pt < PT_BINS_FLOW[-1])
                pt_fl  = pt  [in_range_flow]
                phi_fl = phi [in_range_flow]

                if len(pt_fl) == 0:
                    continue

                ibin_flow = np.searchsorted(PT_BINS_FLOW[1:], pt_fl)

                for ib in range(N_PT_FLOW):
                    sel = ibin_flow == ib
                    if not np.any(sel):
                        continue
                    pt_b  = pt_fl [sel]
                    phi_b = phi_fl[sel]

                    for io, n in enumerate(ORDERS):
                        Qn  = np.sum(np.exp( 1j * n * phi_b))
                        Q2n = np.sum(np.exp(2j * n * phi_b))
                        pQn = np.sum(pt_b * np.exp(1j * n * phi_b))

                        Qn_real [iev, RCUT, ib, io] = Qn.real
                        Qn_imag [iev, RCUT, ib, io] = Qn.imag
                        Q2n_real[iev, RCUT, ib, io] = Q2n.real
                        Q2n_imag[iev, RCUT, ib, io] = Q2n.imag
                        pQn_real[iev, RCUT, ib, io] = pQn.real
                        pQn_imag[iev, RCUT, ib, io] = pQn.imag

        out[name] = {
            'N_pt':     N_pt,      # fine grid (n_ev, n_rap, N_PT)
            'sum_pt':   sum_pt,    # fine grid
            'Qn_real':  Qn_real,   # coarse grid (n_ev, n_rap, N_PT_FLOW, n_ord)
            'Qn_imag':  Qn_imag,
            'Q2n_real': Q2n_real,
            'Q2n_imag': Q2n_imag,
            'pQn_real': pQn_real,
            'pQn_imag': pQn_imag,
            'N_eta':    N_eta,    # shape (n_ev, N_ETA) — new
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: HDF5 output
# ══════════════════════════════════════════════════════════════════════════════

def save_hdf5(filename, summary, hydro_event_id=0):
    """
    Write the per-node Q-vector summary and run metadata to an HDF5 file.

    File structure
    --------------
    /metadata/
        attrs: hydro_event_id, pt_bins, pt_cents, orders, rap_cuts, n_rap_cuts
        species_pdg/
            attrs: one attr per species — int for single species, int32 array for groups
        rap_cuts_detail/
            attrs: cut_0, cut_0_label, cut_1, cut_1_label, ...

    /particles/<species_name>/
        attrs: pdg — int or int32 array of PDG ids
        N_pt     : int32   (n_events, n_rap_cuts, n_pt)
        sum_pt   : float32 (n_events, n_rap_cuts, n_pt)
        Qn_real  : float32 (n_events, n_rap_cuts, n_pt, n_ord)
        Qn_imag  : float32 (n_events, n_rap_cuts, n_pt, n_ord)
        Q2n_real : float32 (n_events, n_rap_cuts, n_pt, n_ord)
        Q2n_imag : float32 (n_events, n_rap_cuts, n_pt, n_ord)
        pQn_real : float32 (n_events, n_rap_cuts, n_pt, n_ord)
        pQn_imag : float32 (n_events, n_rap_cuts, n_pt, n_ord)

    Parameters
    ----------
    filename         : str  — output HDF5 file path
    summary          : dict — output of compute_summary()
    hydro_event_id   : int  — unique integer ID for this hydro event
    """
    with h5py.File(filename, 'w') as f:

        # ── metadata group ─────────────────────────────────────────────────
        meta = f.create_group('metadata')
        meta.attrs['hydro_event_id'] = hydro_event_id
        meta.attrs['pt_bins']      = PT_BINS        # fine
        meta.attrs['pt_cents']     = PT_CENTS       # fine
        meta.attrs['pt_bins_flow'] = PT_BINS_FLOW   # coarse
        meta.attrs['pt_cents_flow']= PT_CENTS_FLOW  # coarse
        meta.attrs['eta_bins']  = ETA_BINS
        meta.attrs['eta_cents'] = ETA_CENTS

        meta.attrs['orders']         = np.array(ORDERS)

        # Rapidity cuts: stored as 2D array of shape (n_rap_cuts, 2)
        # where row i = [rap_min, rap_max] for cut index i.
        rap_array = np.array([RAP_CUTS[i] for i in sorted(RAP_CUTS.keys())],
                             dtype=np.float32)
        meta.attrs['rap_cuts']   = rap_array
        meta.attrs['n_rap_cuts'] = len(RAP_CUTS)

        # Individual cut labels for human readability.
        rap_grp = meta.create_group('rap_cuts_detail')
        for idx, bounds in RAP_CUTS.items():
            rap_grp.attrs[f'cut_{idx}']       = np.array(bounds, dtype=np.float32)
            rap_grp.attrs[f'cut_{idx}_label'] = f'|y| in ({bounds[0]:.1f}, {bounds[1]:.1f})'

        # Species names as a simple string list — safe because all values are strings.
        meta.attrs['species_names'] = list(SPECIES.keys())

        # Species PDG ids stored per-species because SPECIES values are
        # inhomogeneous (int for single species, list of ints for 'charged').
        # Storing list(SPECIES.values()) as a single attr would fail.
        sp_grp = meta.create_group('species_pdg')
        for sp_name, pdg_spec in SPECIES.items():
            if isinstance(pdg_spec, int):
                sp_grp.attrs[sp_name] = pdg_spec
            else:
                sp_grp.attrs[sp_name] = np.array(pdg_spec, dtype=np.int32)

        # ── particle Q-vector data ─────────────────────────────────────────
        for name, data in summary.items():
            grp = f.create_group(f'particles/{name}')

            # Store PDG spec as attribute — int or array depending on species.
            pdg_spec = SPECIES[name]
            if isinstance(pdg_spec, int):
                grp.attrs['pdg'] = pdg_spec
            else:
                grp.attrs['pdg'] = np.array(pdg_spec, dtype=np.int32)

            for key, arr in data.items():
                grp.create_dataset(key, data=arr,
                                   compression='gzip', compression_opts=4)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(events):
    """
    Print basic diagnostics on the first sampling event to verify the
    binary file was read correctly before committing to the full analysis.

    Checks performed:
    - Total particle count in first event
    - Set of unique PDG ids (should include pions, kaons, protons, resonances)
    - Energy range (should be O(0.1) to O(few) GeV for thermal particles)
    - pi+ pole mass (should be ~0.138 GeV — good struct layout check)
    - Mean pT of pi+ (should be ~0.3-0.6 GeV for LHC energies)
    """
    if len(events) == 0:
        print("WARNING: no events found — check binary file path.")
        return

    ev = events[0]
    print("=" * 55)
    print("Sanity check — first sampling event")
    print(f"  N particles        : {len(ev)}")
    print(f"  Unique PDG ids     : {np.unique(ev['pid'])}")

    if len(ev) > 0:
        print(f"  E range (GeV)      : {ev['E'].min():.3f} — {ev['E'].max():.3f}")

        pions = ev[ev['pid'] == 211]
        if len(pions) > 0:
            pt_pi = np.sqrt(pions['px']**2 + pions['py']**2)
            print(f"  pi+ count          : {len(pions)}")
            print(f"  pi+ mass (GeV)     : {pions['mass'][0]:.4f}  (expect 0.1380)")
            print(f"  pi+ <pT> (GeV)     : {pt_pi.mean():.4f}  (expect 0.3—0.6)")

    print(f"  Total events       : {len(events)}")
    print("=" * 55)
