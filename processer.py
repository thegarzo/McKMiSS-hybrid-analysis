import numpy as np
import h5py
import sys

from parameters import *
import kinematics as kn


def inspect_hdf5(filename):
    """Print the full structure of an HDF5 file."""
    with h5py.File(filename, 'r') as f:

        print(f"File: {filename}")
        print("=" * 55)

        def print_tree(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}[GROUP]  /{name}")
                for k, v in obj.attrs.items():
                    print(f"{indent}  attr: {k} = {v}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}[DATA]   /{name}  shape={obj.shape}  dtype={obj.dtype}")

        f.visititems(print_tree)

        # also print top-level attrs
        print("\nTop-level attributes:")
        for k, v in f.attrs.items():
            print(f"  {k} = {v}")

def load_hdf5(filename):
    """
    Load summary and reconstruct RAP_CUTS, PT_BINS, ORDERS from metadata
    so the analysis code doesn't need to know the original global definitions.
    """
    with h5py.File(filename, 'r') as f:
        meta = f['metadata']

        pt_bins  = meta.attrs['pt_bins']
        orders   = list(meta.attrs['orders'])

        # Reconstruct RAP_CUTS dict from the stored 2D array.
        rap_array = meta.attrs['rap_cuts']   # shape (n_rap_cuts, 2)
        rap_cuts  = {i: list(rap_array[i]) for i in range(len(rap_array))}

        # Load particle data.
        summary = {}
        for name in f['particles']:
            grp = f[f'particles/{name}']
            summary[name] = {key: grp[key][:] for key in grp.keys()}

    return summary, pt_bins, orders, rap_cuts

def first_pass_centrality(filenames, irap_cent,
                          charged_key='charged_hadrons'):
    """
    First pass: extract dNch/deta per sampling event for centrality estimation.

    Parameters
    ----------
    filenames     : list of str
    irap_cent     : int — rapidity cut index to use as centrality estimator.
                    Check your metadata rap_cuts to confirm which index
                    corresponds to the window you want, e.g.:
                        0: [-0.8,  0.8]  midrapidity
                        1: [-3.3, -2.1]  backward (ALICE V0C-like)
                        2: [ 2.8,  5.1]  forward  (ALICE V0A-like)
    charged_key   : str — name of charged hadron species in HDF5 file.
                    Defaults to 'charged_hadrons' — check with inspect_hdf5()
                    if unsure.
    """
    filenames = sorted(filenames)

    all_N_ch        = []
    all_file_index  = []
    all_event_index = []
    rap_window      = None
    n_events_per_file = []

    for ifile, fname in enumerate(filenames):
        with h5py.File(fname, 'r') as f:

            # ── validate rapidity window ───────────────────────────────
            rap_cuts = f['metadata'].attrs['rap_cuts']  # shape (n_rap, 2)

            if irap_cent >= len(rap_cuts):
                raise ValueError(
                    f"irap_cent={irap_cent} out of range: file {fname} "
                    f"has {len(rap_cuts)} rapidity cuts:\n{rap_cuts}"
                )

            this_rap_window = list(rap_cuts[irap_cent])

            if rap_window is None:
                rap_window = this_rap_window
                print(f"Centrality estimator : '{charged_key}'")
                print(f"Rapidity window      : [{rap_window[0]:.2f}, "
                      f"{rap_window[1]:.2f}]  (irap={irap_cent})")
            else:
                if not np.allclose(rap_window, this_rap_window, atol=1e-4):
                    raise ValueError(
                        f"RAP_CUTS mismatch at irap={irap_cent}: "
                        f"expected {rap_window}, got {this_rap_window} "
                        f"in {fname}."
                    )

            # ── validate charged species exists ────────────────────────
            path = f'particles/{charged_key}/N_pt'
            if path not in f:
                available = list(f['particles'].keys())
                raise KeyError(
                    f"'{charged_key}' not found in {fname}. "
                    f"Available species: {available}"
                )

            # ── read N_pt for this rapidity cut only ───────────────────
            # Shape: (n_samp, n_rap, n_pt) -> select irap, sum over pT
            N_ch   = f[path][:, irap_cent, :].sum(axis=-1).astype(np.int32)
            n_samp = len(N_ch)

            all_N_ch.append(N_ch)
            all_file_index.append(np.full(n_samp, ifile, dtype=np.int32))
            all_event_index.append(np.arange(n_samp,     dtype=np.int32))
            n_events_per_file.append(n_samp)

        if (ifile + 1) % 50 == 0 or (ifile + 1) == len(filenames):
            print(f"  {ifile+1}/{len(filenames)} files, "
                  f"{sum(n_events_per_file)} sampling events ...")

    # ── concatenate ────────────────────────────────────────────────────
    N_ch_all        = np.concatenate(all_N_ch)
    file_index_all  = np.concatenate(all_file_index)
    event_index_all = np.concatenate(all_event_index)

    # ── normalise to dNch/deta ─────────────────────────────────────────
    # rap_width = rap_max - rap_min (works for any window, symmetric or not)
    rap_min   = rap_window[0]
    rap_max   = rap_window[1]
    rap_width = rap_max - rap_min          # e.g. 5.1 - 2.8 = 2.3
    dNch_deta = N_ch_all.astype(np.float32) / rap_width

    print(f"\nFirst pass complete:")
    print(f"  Files                 : {len(filenames)}")
    print(f"  Total sampling events : {len(N_ch_all)}")
    print(f"  Rapidity window       : [{rap_min:.2f}, {rap_max:.2f}]  "
          f"width = {rap_width:.2f}")
    print(f"  dNch/deta  mean       : {dNch_deta.mean():.1f}")
    print(f"  dNch/deta  min        : {dNch_deta.min():.1f}")
    print(f"  dNch/deta  max        : {dNch_deta.max():.1f}")

    return {
        'dNch_deta':          dNch_deta,
        'N_ch':               N_ch_all,
        'file_index':         file_index_all,
        'event_index':        event_index_all,
        'rap_window':         rap_window,
        'irap_cent':          irap_cent,
        'charged_key':        charged_key,
        'n_files':            len(filenames),
        'n_events_per_file':  n_events_per_file,
        'filenames':          filenames,
    }


def make_centrality_masks(records, bins_percentile):
    """
    Split sampling events into centrality classes using percentile cuts
    on the dNch/deta distribution extracted in the first pass.

    Convention follows standard heavy-ion practice:
        0-5%   = most central   = highest multiplicity
        80-100% = most peripheral = lowest multiplicity

    Parameters
    ----------
    records         : dict — output of first_pass_centrality()
    bins_percentile : list of int — percentile edges,
                      e.g. [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]

    Returns
    -------
    masks : dict {label: np.ndarray of bool, shape (n_total_sampling_events,)}
            e.g. masks['0-5'] selects the 5% most central events
    info  : dict {label: {'dNch_range': [min, max], 'n_events': int}}
    """
    dNch  = records['dNch_deta']
    n_tot = len(dNch)

    # Percentile edges in multiplicity space.
    # 100 - percentile because high multiplicity = central = low percentile number.
    edges = np.percentile(dNch, 100 - np.array(bins_percentile))

    print(f"Centrality definition")
    print(f"  Total sampling events : {n_tot}")
    print(f"  dNch/deta range       : [{dNch.min():.1f}, {dNch.max():.1f}]")
    print(f"  Rapidity window       : {records['rap_window']}")
    print()
    print(f"  {'Class':8s}  {'dNch/deta min':>14s}  {'dNch/deta max':>14s}  {'N events':>10s}  {'Fraction':>8s}")
    print(f"  {'-'*60}")

    masks = {}
    info  = {}

    for i in range(len(bins_percentile) - 1):
        label    = f'{bins_percentile[i]}-{bins_percentile[i+1]}'
        mult_max = edges[i]       # upper multiplicity edge (more central)
        mult_min = edges[i + 1]   # lower multiplicity edge (less central)

        # Edge case: include the maximum value in the most central bin
        # to avoid off-by-one from floating point percentile calculation.
        if i == 0:
            mask = dNch >= mult_min
        else:
            mask = (dNch >= mult_min) & (dNch < mult_max)

        n_sel    = mask.sum()
        fraction = n_sel / n_tot

        masks[label] = mask
        info[label]  = {
            'dNch_range': [float(mult_min), float(mult_max)],
            'n_events':   int(n_sel),
            'fraction':   float(fraction),
        }

        print(f"  {label:8s}  {mult_min:>14.1f}  {mult_max:>14.1f}  "
              f"{n_sel:>10d}  {fraction:>8.3f}")

    # Sanity check: all events should be assigned to exactly one class.
    total_assigned = sum(m.sum() for m in masks.values())
    if total_assigned != n_tot:
        print(f"\n  WARNING: {total_assigned} events assigned but "
              f"{n_tot} total — check for gaps at bin edges.")
    else:
        print(f"\n  All {n_tot} events assigned correctly.")

    return masks, info

def compute_dNch_deta(masks, records, species='charged_hadrons'):
    """
    Compute dNch/deta vs eta per centrality class.
    """
    files    = records['filenames']
    labels   = list(masks.keys())
    acc_Neta = {lab: None for lab in labels}
    n_events = {lab: 0    for lab in labels}
    eta_bins = eta_cents = None
    offset   = 0

    for fname in files:
        with h5py.File(fname, 'r') as f:
            if eta_bins is None:
                eta_bins  = f['metadata'].attrs['eta_bins']
                eta_cents = f['metadata'].attrs['eta_cents']

            N_eta  = f[f'particles/{species}/N_eta'][:]   # (n_samp, N_ETA)
            n_samp = N_eta.shape[0]
            file_slice = slice(offset, offset + n_samp)

            for lab in labels:
                local_mask = masks[lab][file_slice]
                n_sel      = local_mask.sum()
                if n_sel == 0:
                    continue
                if acc_Neta[lab] is None:
                    acc_Neta[lab] = np.zeros(N_eta.shape[1], dtype=np.float64)
                acc_Neta[lab] += N_eta[local_mask].sum(axis=0)
                n_events[lab] += n_sel

        offset += n_samp

    deta    = np.diff(eta_bins)
    results = {}
    for lab in labels:
        n_ev = n_events[lab]
        if n_ev == 0:
            continue
        dNch_deta     = acc_Neta[lab] / (n_ev * deta)
        results[lab]  = {
            'eta_cents':  eta_cents,
            'dNch_deta':  dNch_deta,
            'n_events':   n_ev,
        }

    return results


def compute_spectra(records,masks, irap, species='pi_plus'):
    """
    Compute pT spectra per centrality class.

    Uses N_pt and sum_pt only — no Q-vectors needed.
    Opens each file once and accumulates histograms directly.

    Parameters
    ----------
    records : dict — output of first_pass_centrality(), provides the
              canonical file order that was used to build the masks.
              This guarantees the offset indexing into the global mask
              is consistent with how the mask was constructed.
    masks   : dict {label: boolean np.ndarray} — from make_centrality_masks()
    irap    : int — rapidity window index for the spectrum
    species : str — particle species key in HDF5

    Returns
    -------
    spectra : dict {label: {
                'pt_cents'   : np.ndarray (n_pt,)  — pT bin centres [GeV]
                'dN'         : np.ndarray (n_pt,)  — dN/dydpT / (2pi pT)
                'dN_err'     : np.ndarray (n_pt,)  — statistical error
                'mean_pt'    : float               — <pT> integrated
                'mean_pt_err': float
                'n_events'   : int
                'rap_window' : [rap_min, rap_max]
             }}
    """

    files = records['filenames']

    # accumulators per centrality class
    # we accumulate: sum of N_pt and sum of N_pt^2 (for error)
    # and sum of sum_pt — all shape (n_pt,)
    centralities    = list(masks.keys())
    acc_N           = {cent: None for cent in centralities}  # sum of N_pt over events
    acc_N2          = {cent: None for cent in centralities}  # sum of N_pt^2 for variance
    acc_sumpt       = {cent: None for cent in centralities}  # sum of sum_pt over events
    n_events        = {cent: 0    for cent in centralities}
    pt_bins         = None
    pt_cents        = None
    rap_window      = None

    # running event offset to index into the global mask
    offset = 0

    for fname in files:
        with h5py.File(fname, 'r') as f:

            # read metadata once per file
            if pt_bins is None:
                pt_bins    = f['metadata'].attrs['pt_bins']
                pt_cents   = f['metadata'].attrs['pt_cents']
                rap_cuts   = f['metadata'].attrs['rap_cuts']
                rap_window = list(rap_cuts[irap])

            # load only what we need: N_pt and sum_pt for this species and irap
            # shape (n_samp, n_rap, n_pt) -> select irap immediately
            N_pt   = f[f'particles/{species}/N_pt']  [:, irap, :]  # (n_samp, n_pt)
            sum_pt = f[f'particles/{species}/sum_pt'][:, irap, :]  # (n_samp, n_pt)

            n_samp = N_pt.shape[0]

            # slice of the global mask for this file
            file_slice = slice(offset, offset + n_samp)

            for cent in centralities:
                local_mask = masks[cent][file_slice]  # boolean (n_samp,)
                n_sel      = local_mask.sum()
                if n_sel == 0:
                    continue

                N_sel   = N_pt  [local_mask, :]   # (n_sel, n_pt)
                spt_sel = sum_pt[local_mask, :]   # (n_sel, n_pt)

                # accumulate
                if acc_N[cent] is None:
                    acc_N    [cent] = np.zeros(N_sel.shape[1], dtype=np.float64)
                    acc_N2   [cent] = np.zeros(N_sel.shape[1], dtype=np.float64)
                    acc_sumpt[cent] = np.zeros(N_sel.shape[1], dtype=np.float64)

                acc_N    [cent] += N_sel.sum(axis=0)
                acc_N2   [cent] += (N_sel.astype(np.float64)**2).sum(axis=0)
                acc_sumpt[cent] += spt_sel.sum(axis=0)
                n_events [cent] += n_sel

            offset += n_samp

    # ── compute spectra from accumulators ──────────────────────────────
    dpt      = np.diff(pt_bins)                    # bin widths
    rap_min  = rap_window[0]
    rap_max  = rap_window[1]
    dy       = rap_max - rap_min                   # rapidity window width

    spectra = {}
    for cent in centralities:
        n_ev = n_events[cent]
        if n_ev == 0:
            print(f"  Warning: no events in centrality {cent}%, skipping.")
            continue

        N_total   = acc_N[cent]                     # (n_pt,)
        N2_total  = acc_N2[cent]
        spt_total = acc_sumpt[cent]

        # spectrum: (1/2pi pT) dN/dy dpT averaged over events
        dN_dydpt  = N_total / (n_ev * 2*np.pi * pt_cents * dpt * dy)

        # statistical error from Poisson counting:
        # sigma(dN) = sqrt(N) / (n_ev * 2pi pT dpT dy)
        dN_err    = np.sqrt(N_total) / (n_ev * 2*np.pi * pt_cents * dpt * dy)

        # <pT> integrated over all pT bins
        total_N   = N_total.sum()
        mean_pt   = spt_total.sum() / total_N if total_N > 0 else np.nan

        # error on <pT>: standard error of the mean across events
        # approximated from per-bin variance
        mean_pt_err = np.sqrt(
            np.sum(acc_N2[cent]) / (n_ev**2)
        ) / total_N if total_N > 0 else np.nan

        spectra[cent] = {
            'pt_cents':    pt_cents,
            'dN':          dN_dydpt,
            'dN_err':      dN_err,
            'mean_pt':     float(mean_pt),
            'mean_pt_err': float(mean_pt_err),
            'n_events':    n_ev,
            'rap_window':  rap_window,
        }

        print(f"  {cent:8s}%  n_ev={n_ev:6d}  "
              f"<pT>={mean_pt:.4f} GeV  "
              f"dN/dy|_{{pT=0}}~{dN_dydpt[0]:.2e}")

    return spectra

def compute_flow(masks, records, irap_sig, irap_ref=None,
                 species='pi_plus', ref_species='charged_hadrons'):
    """
    Compute integrated and pT-differential flow vn{2} per centrality class.

    Uses the scalar product method:
        vn{2}(pT) = c2(pT) / sqrt(c2_ref)

    where the reference Q-vector comes from a separate rapidity window
    (irap_ref) to avoid autocorrelations.

    Parameters
    ----------
    masks       : dict {label: boolean np.ndarray} — from make_centrality_masks()
    records     : dict — from first_pass_centrality(), provides file order
    irap_sig    : int  — rapidity index for signal (flow measurement)
    irap_ref    : int  — rapidity index for reference Q-vector.
                  Should be a DIFFERENT window from irap_sig to avoid
                  autocorrelations. If None, uses irap_sig with
                  self-correlation subtraction.
    species     : str  — signal species
    ref_species : str  — reference species for normalisation

    Returns
    -------
    results : dict {label: {
                'pt_cents'   : np.ndarray (n_pt,)
                'vn2_pt'     : np.ndarray (n_pt, n_ord)   pT-differential
                'vn2_pt_err' : np.ndarray (n_pt, n_ord)
                'vn2_int'    : np.ndarray (n_ord,)         integrated
                'vn2_int_err': np.ndarray (n_ord,)
                'c2_ref'     : np.ndarray (n_ord,)
                'n_events'   : int
             }}
    """
    files      = records['filenames']
    same_window = irap_ref is None or irap_ref == irap_sig
    if irap_ref is None:
        irap_ref = irap_sig

    labels = list(masks.keys())

    # accumulators per centrality — all shape (n_pt, n_ord) or (n_ord,)
    # for the scalar product method we need per-event quantities so we
    # store lists and compute cumulants after loading all files
    acc = {lab: {
        'M_sig':    [],   # (n_samp,)        total signal multiplicity
        'M_ref':    [],   # (n_samp,)        total reference multiplicity
        'Qn_sig':   [],   # (n_samp, n_ord)  integrated signal Q-vector
        'Qn_ref':   [],   # (n_samp, n_ord)  integrated reference Q-vector
        'Q2n_ref':  [],   # (n_samp, n_ord)  for vn{4}
        'Qn_sig_pt':[],   # (n_samp, n_pt, n_ord)  pT-differential signal
        'M_sig_pt': [],   # (n_samp, n_pt)   signal multiplicity per bin
    } for lab in labels}

    n_events  = {lab: 0 for lab in labels}
    pt_cents  = None
    orders    = None
    offset    = 0

    print(f"Computing flow: {species} (sig irap={irap_sig}) "
          f"ref: {ref_species} (irap={irap_ref})")

    for fname in files:
        with h5py.File(fname, 'r') as f:

            if pt_cents is None:
                pt_cents = f['metadata'].attrs['pt_cents_flow']  # coarse grid for flow
                orders   = list(f['metadata'].attrs['orders'])
                n_pt     = len(pt_cents)
                n_ord    = len(orders)

            # ── signal: pT-differential ────────────────────────────────
            # shape (n_samp, n_pt, n_ord)
            Qn_sig_r_pt = f[f'particles/{species}/Qn_real'][:, irap_sig, :, :]
            Qn_sig_i_pt = f[f'particles/{species}/Qn_imag'][:, irap_sig, :, :]
            M_sig_pt    = f[f'particles/{species}/N_pt']   [:, irap_sig, :]

            # integrated signal: sum over pT bins -> (n_samp, n_ord)
            Qn_sig_r = Qn_sig_r_pt.sum(axis=1)
            Qn_sig_i = Qn_sig_i_pt.sum(axis=1)
            M_sig    = M_sig_pt.sum(axis=1)   # (n_samp,)

            # ── reference: integrated only ─────────────────────────────
            Qn_ref_r  = f[f'particles/{ref_species}/Qn_real'] [:, irap_ref, :, :].sum(axis=1)
            Qn_ref_i  = f[f'particles/{ref_species}/Qn_imag'] [:, irap_ref, :, :].sum(axis=1)
            Q2n_ref_r = f[f'particles/{ref_species}/Q2n_real'][:, irap_ref, :, :].sum(axis=1)
            Q2n_ref_i = f[f'particles/{ref_species}/Q2n_imag'][:, irap_ref, :, :].sum(axis=1)
            M_ref     = f[f'particles/{ref_species}/N_pt']    [:, irap_ref, :].sum(axis=1)

            n_samp = M_sig.shape[0]
            file_slice = slice(offset, offset + n_samp)

            for lab in labels:
                local_mask = masks[lab][file_slice]
                n_sel      = local_mask.sum()
                if n_sel == 0:
                    continue

                acc[lab]['M_sig'].append(M_sig        [local_mask])
                acc[lab]['M_ref'].append(M_ref        [local_mask])
                acc[lab]['Qn_sig'].append(
                    Qn_sig_r[local_mask] + 1j * Qn_sig_i[local_mask])
                acc[lab]['Qn_ref'].append(
                    Qn_ref_r[local_mask] + 1j * Qn_ref_i[local_mask])
                acc[lab]['Q2n_ref'].append(
                    Q2n_ref_r[local_mask] + 1j * Q2n_ref_i[local_mask])
                acc[lab]['Qn_sig_pt'].append(
                    Qn_sig_r_pt[local_mask] + 1j * Qn_sig_i_pt[local_mask])
                acc[lab]['M_sig_pt'].append(M_sig_pt[local_mask])
                n_events[lab] += n_sel

            offset += n_samp

    # ── compute cumulants from accumulated per-event arrays ────────────
    results = {}

    for lab in labels:
        n_ev = n_events[lab]
        if n_ev == 0:
            print(f"  Warning: no events in centrality {lab}%, skipping.")
            continue

        # concatenate all selected events
        M_sig    = np.concatenate(acc[lab]['M_sig']).astype(float)    # (n_ev,)
        M_ref    = np.concatenate(acc[lab]['M_ref']).astype(float)    # (n_ev,)
        Qn_sig   = np.concatenate(acc[lab]['Qn_sig'],   axis=0)       # (n_ev, n_ord)
        Qn_ref   = np.concatenate(acc[lab]['Qn_ref'],   axis=0)       # (n_ev, n_ord)
        Q2n_ref  = np.concatenate(acc[lab]['Q2n_ref'],  axis=0)       # (n_ev, n_ord)
        Qn_sig_pt= np.concatenate(acc[lab]['Qn_sig_pt'],axis=0)       # (n_ev, n_pt, n_ord)
        M_sig_pt = np.concatenate(acc[lab]['M_sig_pt'], axis=0)       # (n_ev, n_pt)

        ok = (M_ref >= 4) & (M_sig >= 1)

        vn2_int     = np.full(n_ord, np.nan)
        vn2_int_err = np.full(n_ord, np.nan)
        vn2_pt      = np.full((n_pt, n_ord), np.nan)
        vn2_pt_err  = np.full((n_pt, n_ord), np.nan)
        c2_ref      = np.full(n_ord, np.nan)

        for io, n in enumerate(orders):
            Qs  = Qn_sig [ok, io]   # signal Q-vector
            Qr  = Qn_ref [ok, io]   # reference Q-vector
            Q2r = Q2n_ref[ok, io]
            Ms  = M_sig  [ok]
            Mr  = M_ref  [ok]

            # ── reference c2{2} ────────────────────────────────────────
            # standard two-particle cumulant from reference particles
            c2r_vec  = (np.abs(Qr)**2 - Mr) / (Mr * (Mr - 1))
            c2r_mean = c2r_vec.mean()
            c2_ref[io] = c2r_mean

            if c2r_mean <= 0:
                print(f"  Warning: c2_ref <= 0 for n={n}, centrality {lab}")
                continue

            ref_norm = np.sqrt(c2r_mean)

            # ── integrated vn{2} ──────────────────────────────────────
            if same_window:
                # same rapidity window: subtract self-correlations
                num_int = (Qs * np.conj(Qr)).real - Ms
                den_int = Ms * (Mr - 1)
            else:
                # different windows: no self-correlations
                num_int = (Qs * np.conj(Qr)).real
                den_int = Ms * Mr

            ok_int = den_int > 0
            c2_int = (num_int[ok_int] / den_int[ok_int]).mean()
            vn2_int[io] = c2_int / ref_norm

            # bootstrap error
            n_ok   = ok_int.sum()
            rat    = num_int[ok_int] / den_int[ok_int]
            boot   = np.array([
                np.mean(rat[np.random.randint(0, n_ok, n_ok)]) / ref_norm
                for _ in range(200)
            ])
            vn2_int_err[io] = np.std(boot)

            # ── pT-differential vn{2}(pT) ─────────────────────────────
            for ib in range(n_pt):
                Qb = Qn_sig_pt[ok, ib, io]
                Mb = M_sig_pt [ok, ib].astype(float)

                ok_b = Mb >= 1
                if ok_b.sum() < 10:
                    continue

                Qb_ok = Qb[ok_b]
                Qr_ok = Qr[ok_b]
                Mr_ok = Mr[ok_b]
                Mb_ok = Mb[ok_b]

                if same_window:
                    num = (Qb_ok * np.conj(Qr_ok)).real - Mb_ok
                    den = Mb_ok * (Mr_ok - 1)
                else:
                    num = (Qb_ok * np.conj(Qr_ok)).real
                    den = Mb_ok * Mr_ok

                ok_den = den > 0
                if ok_den.sum() < 10:
                    continue

                rat_pt = num[ok_den] / den[ok_den]
                c2_pt  = rat_pt.mean()
                vn2_pt[ib, io] = c2_pt / ref_norm

                # bootstrap error per pT bin
                n_d  = ok_den.sum()
                boot = np.array([
                    np.mean(rat_pt[np.random.randint(0, n_d, n_d)]) / ref_norm
                    for _ in range(200)
                ])
                vn2_pt_err[ib, io] = np.std(boot)

        results[lab] = {
            'pt_cents':    pt_cents,
            'orders':      orders,
            'vn2_int':     vn2_int,
            'vn2_int_err': vn2_int_err,
            'vn2_pt':      vn2_pt,
            'vn2_pt_err':  vn2_pt_err,
            'c2_ref':      c2_ref,
            'n_events':    n_ev,
        }

        print(f"  {lab:8s}%  n_ev={n_ev:6d}  "
              + "  ".join(f"v{n}{{2}}={vn2_int[io]:.4f}±{vn2_int_err[io]:.4f}"
                          for io, n in enumerate(orders)))

    return results

def compute_flow_cumulants(masks, records, 
                           irap_mid, irap_subA, irap_subB,
                           species='pi_plus',
                           ref_species='charged_hadrons'):
    """
    Compute flow cumulants using the direct Q-vector method.

    Observables computed
    --------------------
    v2 {2}     : standard two-particle cumulant, both particles from irap_mid.
                Self-correlations subtracted analytically.
                Reference: Bilandzic et al., PRC 83, 044913 (2011).

    v2{2|A|B} : two-particle cumulant with rapidity gap sub-event method.
                Particle A from irap_subA, particle B from irap_subB.
                No self-correlations by construction (different eta windows).
                Reference: Jia et al., PRL 116, 172301 (2016).

    v2{2}(pT) : pT-differential, signal from irap_mid,
                reference Q-vector from irap_subB (rapidity gap).

    v2{4}     : four-particle cumulant, all from irap_mid.
                Reference: Bilandzic et al., PRC 83, 044913 (2011).

    Parameters
    ----------
    masks       : dict {label: boolean np.ndarray}
    records     : dict — from first_pass_centrality()
    irap_mid    : int  — midrapidity window index (signal + standard cumulants)
    irap_subA   : int  — sub-event A rapidity index
    irap_subB   : int  — sub-event B rapidity index
    species     : str  — signal species for pT-differential
    ref_species : str  — species for reference Q-vectors

    Returns
    -------
    results : dict {centrality_label: {
                'orders'       : list of int
                'pt_cents'     : np.ndarray (n_pt_flow,)
                'v2_2'         : np.ndarray (n_ord,)   standard c2{2}
                'v2_2_err'     : np.ndarray (n_ord,)
                'v2_2sub'      : np.ndarray (n_ord,)   sub-event c2{2|AB}
                'v2_2sub_err'  : np.ndarray (n_ord,)
                'v2_4'         : np.ndarray (n_ord,)   c2{4}
                'v2_4_err'     : np.ndarray (n_ord,)
                'vn2_pt'       : np.ndarray (n_pt, n_ord)  pT-differential
                'vn2_pt_err'   : np.ndarray (n_pt, n_ord)
                'n_events'     : int
             }}
    """
    files    = records['filenames']
    labels   = list(masks.keys())
    n_events = {lab: 0 for lab in labels}
    pt_cents = orders = None
    offset   = 0

    # per-event accumulators — store raw per-event quantities,
    # compute cumulants after loading all files
    acc = {lab: {
        # standard cumulant: midrapidity
        'M_mid':     [],   # (n_ev,)
        'Qn_mid':    [],   # (n_ev, n_ord)
        'Q2n_mid':   [],   # (n_ev, n_ord)  for v2{4}
        # sub-event A and B
        'Qn_subA':   [],   # (n_ev, n_ord)
        'M_subA':    [],
        'Qn_subB':   [],   # (n_ev, n_ord)
        'M_subB':    [],
        # pT-differential: signal at midrapidity (coarse grid)
        'Qn_mid_pt': [],   # (n_ev, n_pt_flow, n_ord)
        'M_mid_pt':  [],   # (n_ev, n_pt_flow)
    } for lab in labels}

    print(f"Computing flow cumulants:")
    print(f"  Signal species  : {species}  irap_mid={irap_mid}")
    print(f"  Sub-event A     : {ref_species}  irap_subA={irap_subA}")
    print(f"  Sub-event B     : {ref_species}  irap_subB={irap_subB}")

    for fname in files:
        with h5py.File(fname, 'r') as f:
            rap_cuts = f['metadata'].attrs['rap_cuts']
            print("rap_cuts in FILE:")
            for i, cut in enumerate(rap_cuts):
                print(f"  irap={i}  [{cut[0]:.2f}, {cut[1]:.2f}]")
            NA = f['particles/charged_hadrons/N_pt'][:, irap_subA, :].sum(axis=-1)
            NB = f['particles/charged_hadrons/N_pt'][:, irap_subB, :].sum(axis=-1)
            
            print(f"Sub-event A  mean N = {NA.mean():.2f}  max = {NA.max():.0f}")
            print(f"Sub-event B  mean N = {NB.mean():.2f}  max = {NB.max():.0f}")

            if pt_cents is None:
                pt_cents  = f['metadata'].attrs['pt_cents_flow']
                orders    = list(f['metadata'].attrs['orders'])
                n_pt      = len(pt_cents)
                n_ord     = len(orders)

            # ── midrapidity signal ─────────────────────────────────────
            # integrated: sum over pT bins -> (n_samp, n_ord)
            Qn_mid_r  = f[f'particles/{species}/Qn_real'] [:, irap_mid, :, :].sum(axis=1)
            Qn_mid_i  = f[f'particles/{species}/Qn_imag'] [:, irap_mid, :, :].sum(axis=1)
            Q2n_mid_r = f[f'particles/{species}/Q2n_real'][:, irap_mid, :, :].sum(axis=1)
            Q2n_mid_i = f[f'particles/{species}/Q2n_imag'][:, irap_mid, :, :].sum(axis=1)
            M_mid     = f[f'particles/{species}/N_pt']    [:, irap_mid, :].sum(axis=1)

            # pT-differential: keep pT axis -> (n_samp, n_pt_flow, n_ord)
            Qn_mid_pt_r = f[f'particles/{species}/Qn_real'][:, irap_mid, :, :]
            Qn_mid_pt_i = f[f'particles/{species}/Qn_imag'][:, irap_mid, :, :]
            M_mid_pt    = f[f'particles/{species}/N_pt']   [:, irap_mid, :, ]

            # ── sub-event A ────────────────────────────────────────────
            Qn_subA_r = f[f'particles/{ref_species}/Qn_real'][:, irap_subA, :, :].sum(axis=1)
            Qn_subA_i = f[f'particles/{ref_species}/Qn_imag'][:, irap_subA, :, :].sum(axis=1)
            M_subA    = f[f'particles/{ref_species}/N_pt']   [:, irap_subA, :].sum(axis=1)

            # ── sub-event B ────────────────────────────────────────────
            Qn_subB_r = f[f'particles/{ref_species}/Qn_real'][:, irap_subB, :, :].sum(axis=1)
            Qn_subB_i = f[f'particles/{ref_species}/Qn_imag'][:, irap_subB, :, :].sum(axis=1)
            M_subB    = f[f'particles/{ref_species}/N_pt']   [:, irap_subB, :].sum(axis=1)

            n_samp     = M_mid.shape[0]
            file_slice = slice(offset, offset + n_samp)

            for lab in labels:
                local_mask = masks[lab][file_slice]
                if local_mask.sum() == 0:
                    continue

                acc[lab]['M_mid'].append(M_mid[local_mask])
                acc[lab]['Qn_mid'].append(
                    Qn_mid_r[local_mask] + 1j * Qn_mid_i[local_mask])
                acc[lab]['Q2n_mid'].append(
                    Q2n_mid_r[local_mask] + 1j * Q2n_mid_i[local_mask])
                acc[lab]['Qn_subA'].append(
                    Qn_subA_r[local_mask] + 1j * Qn_subA_i[local_mask])
                acc[lab]['M_subA'].append(M_subA[local_mask])
                acc[lab]['Qn_subB'].append(
                    Qn_subB_r[local_mask] + 1j * Qn_subB_i[local_mask])
                acc[lab]['M_subB'].append(M_subB[local_mask])
                acc[lab]['Qn_mid_pt'].append(
                    Qn_mid_pt_r[local_mask] + 1j * Qn_mid_pt_i[local_mask])
                acc[lab]['M_mid_pt'].append(M_mid_pt[local_mask])
                n_events[lab] += local_mask.sum()

            offset += n_samp

    # ── compute cumulants ──────────────────────────────────────────────
    results = {}

    for lab in labels:
        n_ev = n_events[lab]
        if n_ev == 0:
            print(f"  Warning: no events in {lab}%, skipping.")
            continue

        # concatenate
        M_mid    = np.concatenate(acc[lab]['M_mid']).astype(float)
        Qn_mid   = np.concatenate(acc[lab]['Qn_mid'],    axis=0)  # (n_ev, n_ord)
        Q2n_mid  = np.concatenate(acc[lab]['Q2n_mid'],   axis=0)
        Qn_subA  = np.concatenate(acc[lab]['Qn_subA'],   axis=0)
        M_subA   = np.concatenate(acc[lab]['M_subA']).astype(float)
        Qn_subB  = np.concatenate(acc[lab]['Qn_subB'],   axis=0)
        M_subB   = np.concatenate(acc[lab]['M_subB']).astype(float)
        Qn_mid_pt= np.concatenate(acc[lab]['Qn_mid_pt'], axis=0)  # (n_ev, n_pt, n_ord)
        M_mid_pt = np.concatenate(acc[lab]['M_mid_pt'],  axis=0)  # (n_ev, n_pt)

        v2_2     = np.full(n_ord, np.nan)
        v2_2_err = np.full(n_ord, np.nan)
        v2_2sub  = np.full(n_ord, np.nan)
        v2_2sub_err = np.full(n_ord, np.nan)
        v2_4     = np.full(n_ord, np.nan)
        v2_4_err = np.full(n_ord, np.nan)
        vn2_pt   = np.full((n_pt, n_ord), np.nan)
        vn2_pt_err = np.full((n_pt, n_ord), np.nan)

        for io, n in enumerate(orders):
            Qm  = Qn_mid [:, io]
            Q2m = Q2n_mid[:, io]
            QA  = Qn_subA[:, io]
            QB  = Qn_subB[:, io]
            M   = M_mid

            # ── standard c2{2} ─────────────────────────────────────
            # uses only midrapidity particles
            # self-correlation subtraction: -M removes i==j pairs
            # denominator M*(M-1) counts all distinct pairs
            # Ref: Bilandzic PRC 83, 044913 (2011) Eq.(6)
            ok2  = M >= 2
            c2_vec  = (np.abs(Qm[ok2])**2 - M[ok2]) / (M[ok2] * (M[ok2] - 1))
            c2_mean = c2_vec.mean()
            v2_2[io]     = np.sqrt(max(c2_mean, 0.0))
            v2_2_err[io] = _bootstrap_v2(c2_vec, kind='2')

            # ── sub-event c2{2|AB} ─────────────────────────────────
            # A from irap_subA, B from irap_subB — different eta windows
            # so NO self-correlations, denominator is simply M_A * M_B
            # Ref: Jia PRL 116, 172301 (2016)
            ok_sub = (M_subA >= 1) & (M_subB >= 1)
            c2sub_vec = (
                (QA[ok_sub] * np.conj(QB[ok_sub])).real
                / (M_subA[ok_sub] * M_subB[ok_sub])
            )
            c2sub_mean  = c2sub_vec.mean()
            v2_2sub[io]     = np.sqrt(max(c2sub_mean, 0.0))
            v2_2sub_err[io] = _bootstrap_v2(c2sub_vec, kind='2')

            # ── c2{4} ──────────────────────────────────────────────
            # four-particle cumulant from midrapidity
            # Ref: Bilandzic PRC 83, 044913 (2011) Eq.(13)
            ok4 = M >= 4
            Qm4   = Qm [ok4]
            Q2m4  = Q2m[ok4]
            M4    = M  [ok4]

            term1    = np.abs(Qm4)**4
            term2    = np.abs(Q2m4)**2
            term3    = 2.0 * (Q2m4 * np.conj(Qm4)**2).real
            term4    = 4.0 * (M4 - 2) * np.abs(Qm4)**2
            term5    = 2.0 * M4 * (M4 - 3) * (M4 - 1)
            denom4   = M4 * (M4 - 1) * (M4 - 2) * (M4 - 3)
            four_vec = (term1 + term2 - term3 - term4 + term5) / denom4
            c4_mean  = four_vec.mean() - 2.0 * c2_mean**2
            v2_4[io]     = (-c4_mean)**0.25 if c4_mean < 0 else np.nan
            v2_4_err[io] = _bootstrap_v2(c2_vec, kind='4',
                                         four_vec=four_vec,
                                         c2_mean=c2_mean)

            # ── pT-differential v2{2}(pT) ──────────────────────────
            # signal: species at irap_mid per pT bin
            # reference: sub-event B (rapidity gap -> no self-correlations)
            # vn(pT) = <Re(Qn_sig(pT) * Qn_refB*)> / (M_sig(pT) * M_refB)
            #          ─────────────────────────────────────────────────────
            #                        sqrt( c2_subAB )
            # normalise by sub-event cumulant for consistency
            ref_norm = np.sqrt(max(c2sub_mean, 0.0))
            if ref_norm <= 0:
                continue

            for ib in range(n_pt):
                Qb = Qn_mid_pt[:, ib, io]
                Mb = M_mid_pt [:, ib].astype(float)
                ok_b = (Mb >= 1) & (M_subB >= 1)
                if ok_b.sum() < 10:
                    continue

                num    = (Qb[ok_b] * np.conj(QB[ok_b])).real
                den    = Mb[ok_b] * M_subB[ok_b]
                ok_den = den > 0
                if ok_den.sum() < 10:
                    continue

                rat = num[ok_den] / den[ok_den]
                vn2_pt[ib, io]     = rat.mean() / ref_norm
                vn2_pt_err[ib, io] = _bootstrap_rat(rat, ref_norm)

        results[lab] = {
            'orders':      orders,
            'pt_cents':    pt_cents,
            'vn_2':        v2_2,        # was 'v2_2'
            'vn_2_err':    v2_2_err,    # was 'v2_2_err'
            'vn_2sub':     v2_2sub,     # was 'v2_2sub'
            'vn_2sub_err': v2_2sub_err, # was 'v2_2sub_err'
            'vn_4':        v2_4,        # was 'v2_4'
            'vn_4_err':    v2_4_err,    # was 'v2_4_err'
            'vn_2_pt':     vn2_pt,      # was 'vn2_pt'
            'vn_2_pt_err': vn2_pt_err,  # was 'vn2_pt_err'
            'n_events':    n_ev,
        }

        print(f"  {lab:8s}%  n={n_ev:5d}  "
              + "  ".join(
                  f"v{n}{{2}}={v2_2[io]:.4f}  "
                  f"v{n}{{2|AB}}={v2_2sub[io]:.4f}  "
                  f"v{n}{{4}}={v2_4[io]:.4f}"
                  for io, n in enumerate(orders)))

    return results


def _bootstrap_v2(c2_vec, kind='2', n_boot=200, four_vec=None, c2_mean=None):
    """Bootstrap error on vn{2} or vn{4}."""
    n = len(c2_vec)
    boot = np.zeros(n_boot)
    for ib in range(n_boot):
        idx  = np.random.randint(0, n, n)
        c2_b = c2_vec[idx].mean()
        if kind == '2':
            boot[ib] = np.sqrt(max(c2_b, 0.0))
        elif kind == '4' and four_vec is not None:
            c4_b = four_vec[idx].mean() - 2.0 * c2_b**2
            boot[ib] = (-c4_b)**0.25 if c4_b < 0 else np.nan
    return np.nanstd(boot)


def _bootstrap_rat(rat, ref_norm, n_boot=200):
    """Bootstrap error on pT-differential vn{2}."""
    n    = len(rat)
    boot = np.array([
        rat[np.random.randint(0, n, n)].mean() / ref_norm
        for _ in range(n_boot)
    ])
    return np.nanstd(boot)