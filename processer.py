import numpy as np
import h5py
import sys

from parameters import *
import kinematics as kn
import auxiliaries as aux

NBoot = 200
def find_ersatz(filenames, ifile):
    found=False
    ersatz=''
    for iers in np.arange(ifile+1):
        if filenames[ifile-iers]!="empty_event":
            ersatz=filenames[ifile-iers]
            found =True 
            break
    iers=0
    while found ==False:
        iers+=1 
        if filenames[ifile+iers]!="empty_event":
            ersatz=filenames[ifile-iers]
            found =True 
    return ersatz
            
def ratio_error(A,dA,B,dB):
    R=A/B
    err2 = np.power(dA/B,2.0) +  np.power(R*dB/B,2.0) 
    return R, np.sqrt(err2)

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
        if fname=="empty_event":
            # find the last available
            fname_ersatz=find_ersatz(filenames,ifile)
            
            with h5py.File(fname_ersatz, 'r') as f:

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

                    all_N_ch.append(0*N_ch)
                    all_file_index.append(np.full(n_samp, ifile, dtype=np.int32))
                    all_event_index.append(np.arange(n_samp,     dtype=np.int32))
                    n_events_per_file.append(n_samp)

        else:
            try:
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

            except Exception as e:
                print(f"Error opening file: {fname}")
                print(f"Error message: {e}")
                # optionally continue to next file or raise
                continue
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
    print(f" ")

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

# def compute_dNch_deta(masks, records, species='charged_hadrons'):
#     """
#     Compute dNch/deta vs eta per centrality class.
#     """
#     files    = records['filenames']
#     labels   = list(masks.keys())
#     acc_Neta = {lab: None for lab in labels}
#     n_events = {lab: 0    for lab in labels}
#     eta_bins = eta_cents = None
#     offset   = 0

#     for fname in files:
#         with h5py.File(fname, 'r') as f:
#             if eta_bins is None:
#                 eta_bins  = f['metadata'].attrs['eta_bins']
#                 eta_cents = f['metadata'].attrs['eta_cents']

#             N_eta  = f[f'particles/{species}/N_eta'][:]   # (n_samp, N_ETA)
#             n_samp = N_eta.shape[0]
#             file_slice = slice(offset, offset + n_samp)

#             for lab in labels:
#                 local_mask = masks[lab][file_slice]
#                 n_sel      = local_mask.sum()
#                 if n_sel == 0:
#                     continue
#                 if acc_Neta[lab] is None:
#                     acc_Neta[lab] = np.zeros(N_eta.shape[1], dtype=np.float64)
#                 acc_Neta[lab] += N_eta[local_mask].sum(axis=0)
#                 n_events[lab] += n_sel

#         offset += n_samp

#     deta    = np.diff(eta_bins)
#     results = {}
#     for lab in labels:
#         n_ev = n_events[lab]
#         if n_ev == 0:
#             continue
#         dNch_deta     = acc_Neta[lab] / (n_ev * deta)
#         results[lab]  = {
#             'eta_cents':  eta_cents,
#             'dNch_deta':  dNch_deta,
#             'n_events':   n_ev,
#         }

#     return results

def compute_dNch_deta(masks, records, species='charged_hadrons'):
    """
    Compute dNch/deta vs eta per centrality class, with statistical errors.
    
    Errors are computed using Poisson statistics:
    - Error on total count N = sqrt(N)
    - Error on dNch/deta = sqrt(N_total) / (n_ev * deta)
    
    This accounts for the natural statistical fluctuations in particle counts.
    """
    files    = records['filenames']
    labels   = list(masks.keys())
    # print('label',labels)
    acc_Neta = {lab: None for lab in labels}     # sum of N_eta
    n_events = {lab: 0    for lab in labels}
    eta_bins = eta_cents = None
    offset   = 0

    for ifile, fname in enumerate(files):
        if fname == "empty_event":
            fname_ersatz=find_ersatz(files,ifile)
            with h5py.File(fname_ersatz, 'r') as f:
                if eta_bins is None:
                    eta_bins  = f['metadata'].attrs['eta_bins']
                    eta_cents = f['metadata'].attrs['eta_cents']

                N_eta  = 0*f[f'particles/{species}/N_eta'][:]   # (n_samp, N_ETA)
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
        else: 
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
        # Mean dNch/deta per event
        dNch_deta     = acc_Neta[lab] / (n_ev * deta)
        # Statistical error using Poisson statistics
        # sqrt(N_total) / (n_ev * deta) where N_total is the count in each eta bin
        dNch_deta_err = np.sqrt(acc_Neta[lab]) / (n_ev * deta)
        
        results[lab]  = {
            'eta_cents':  eta_cents,
            'dNch_deta':  dNch_deta,
            'dNch_deta_err': dNch_deta_err,
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

    centralities  = list(masks.keys())
    n_events = {cent: 0 for cent in centralities}

    # per-event accumulators — store raw per-event quantities,
    # compute cumulants after loading all files
    # Let's fill the reference class
    acc = {cent: {
        # standard cumulant: midrapidity
        'N_tot':    [],   # (n_ev, n_samp)
        'N_pt':     [],   # (n_ev, n_samp, n_pt)
        # 'sum_pt':   [],   # (n_ev, n_ord)
        'avg_pt':   [],   # (n_ev, n_ord)  for v2{4}
        } for cent in centralities}
    # now the signal class

    # accumulators per centrality class
    # we accumulate: sum of N_pt and sum of N_pt^2 (for error)
    # and sum of sum_pt — all shape (n_pt,)
    # acc_N           = {cent: None for cent in centralities}  # sum of N_pt over events
    # acc_N2          = {cent: None for cent in centralities}  # sum of N_pt^2 for variance
    # acc_sumpt       = {cent: None for cent in centralities}  # sum of sum_pt over events
    # acc_sumpt2      = {cent: None for cent in centralities}  # sum of sum_pt over events
    # acc_Ns          = {cent: None for cent in centralities}  # sum of particles (no pt bins)
    # acc_Ns2         = {cent: None for cent in centralities}  # 
    # n_events        = {cent: 0    for cent in centralities}
    pt_bins         = None
    pt_cents        = None
    rap_window      = None

    # running event offset to index into the global mask
    offset = 0

    

    for ifile, fname in enumerate(files):
        if fname == "empty_event":
            fname_ersatz=find_ersatz(files,ifile)
        else: 
            fname_ersatz=fname   
        
            with h5py.File(fname_ersatz, 'r') as f:

                # read metadata once per file
                if pt_bins is None:
                    pt_bins    = f['metadata'].attrs['pt_bins']
                    pt_cents   = f['metadata'].attrs['pt_cents']
                    rap_cuts   = f['metadata'].attrs['rap_cuts']
                    rap_window = list(rap_cuts[irap])

                # load only what we need: N_pt and sum_pt for this species and irap
                # shape (n_samp, n_rap, n_pt) -> select irap immediately
                if fname == "empty_event":
                    N_pt   = 0*f[f'particles/{species}/N_pt']  [:, irap, :]  # (n_samp, n_pt)
                    sum_pt = 0*f[f'particles/{species}/sum_pt'][:, irap, :]  # (n_samp, n_pt)
                else: 
                    N_pt   = f[f'particles/{species}/N_pt']  [:, irap, :]  # (n_samp, n_pt)
                    sum_pt = f[f'particles/{species}/sum_pt'][:, irap, :]  # (n_samp, n_pt)
                

                n_samp = N_pt.shape[0]

                # slice of the global mask for this file
                file_slice = slice(offset, offset + n_samp)

                for cent in centralities:
                    local_mask = masks[cent][file_slice]  # boolean (n_samp,)
                    # remember n_sel is events
                    n_sel      = local_mask.sum()
                    
                    if n_sel == 0:
                        continue

                    N_sel   = N_pt[local_mask, :]   # (n_sel, n_pt)
                    N_tot   = N_sel.sum(axis=1)   # (n_sel, n_pt)
                    spt_sel = sum_pt[local_mask, :]   # (n_sel, n_pt)
                    
                    N_sel_tot= N_sel[:9].sum(axis=1)
                    spt_sel_tot= spt_sel[:9].sum(axis=1)

    
                    # print
                    # accumulate
                    # print(cent, " test")
                    # print(spt_sel.shape , N_sel.shape)
                    # print(spt_sel_tot.shape , N_tot.shape)
                    
                    acc[cent]['N_pt'].append(N_sel)
                    acc[cent]['N_tot'].append(N_tot)
                    
                    avg_pt_t = np.where(N_sel_tot > 0, spt_sel_tot / N_sel_tot, 0)
                    acc[cent]['avg_pt'].append(avg_pt_t)
                    # if N_tot>0:
                    #     acc['Ref'][cent]['avg_pt'].append(spt_sel_tot/float(N_tot))
                    # else:
                    #     acc['Ref'][cent]['avg_pt'].append(np.zeros(len(N_tot)))
                
                    n_events [cent] += n_sel
                    # print(species, cent, n_sel,avg_pt_events)
                offset += n_samp

    # ── compute spectra from accumulators ──────────────────────────────
    dpt      = np.diff(pt_bins)                    # bin widths
    rap_min  = rap_window[0]
    rap_max  = rap_window[1]
    deta       = rap_max - rap_min                   # rapidity window width
    # print(pt_cents)
    spectra = {}
    for cent in centralities:
        # print(acc_sumpt[cent],n_events[cent])
        n_ev = n_events[cent]
        if n_ev == 0:
            print(f"  Warning: no events in centrality {cent}%, skipping.")
            continue
# = np.concatenate(acc['Ref'][lab]['M_subA']).astype(float)
        # print(len(acc['Ref'][cent]['N_tot']), acc['Ref'][cent]['N_tot'][0].shape )
        
        # N_total_ev   =  np.concatenate(acc['Ref'][cent]['N_tot'], axis=0) # (nevs,n_pt)
        # N_pt_ev   =  np.concatenate(acc['Ref'][cent]['N_pt'], axis=0) # (nevs,n_pt)
        # avg_pt_ev   =  np.concatenate(acc['Ref'][cent]['avg_pt'], axis=0) # (nevs,n_pt)

        Npacks=len(acc[cent]['N_tot'])
        Npt= (acc[cent]['N_pt'][0].shape)[1]
        # print(Npacks,Npt)

        njs = np.array([len(event) for event in acc[cent]['N_tot'] ] )
        Nev_tot=np.sum(njs)
        weights = njs/Nev_tot
        # print(weights)
        # print(Nev_tot,njs) 

        N_total_ev   = np.zeros(Npacks)
        N_total_ev_err   = np.zeros(Npacks)
        N_pt_ev   = np.zeros((Npacks,Npt)) 
        N_pt_ev_err   = np.zeros((Npacks,Npt)) # acc['Ref'][cent]['N_pt'] # (nevs,n_pt)
        avg_pt_ev   = np.zeros(Npacks) # (nevs,n_pt)
        avg_pt_ev_err   = np.zeros(Npacks) #acc['Ref'][cent]['avg_pt'], axis=0) # (nevs,n_pt)

        for i in np.arange(Npacks):
            N_total_ev[i] = np.mean(acc[cent]['N_tot'][i])
            N_total_ev_err[i]   = bootstrap_error_simple(acc[cent]['N_tot'][i])
            N_pt_ev[i]   = np.mean(acc[cent]['N_pt'][i],axis=0)
            N_pt_ev_err[i]   = bootstrap_error(acc[cent]['N_pt'][i],axis=0)
            avg_pt_ev[i]   = np.mean(acc[cent]['avg_pt'][i]) # (nevs,n_pt)
            avg_pt_ev_err[i]  = bootstrap_error_simple(acc[cent]['avg_pt'][i]) #acc['Ref'][cent]['avg_pt'], axis=0) # (nevs,n_pt)

        # print(N_total_ev.shape)
        # print(N_pt_ev.shape)
        # print(avg_pt_ev.shape)
        # exit()
        # dNdeta averaged over events
        dN_eta_ev = N_total_ev / deta
        dN_eta_ev_err = N_total_ev_err / deta
        dN_deta  =  np.sum(weights*dN_eta_ev) 
        dN_deta_err =np.sqrt( np.sum(np.power(weights*dN_eta_ev_err,2.)) )

        # spectrum: (1/2pi pT) dN/deta dpT averaged over events
        dN_detadpt_ev =N_pt_ev/  (2*np.pi * pt_cents[np.newaxis,:] * dpt[np.newaxis,:] * deta)
        dN_detadpt_ev_err =N_pt_ev/  (2*np.pi * pt_cents[np.newaxis,:] * dpt[np.newaxis,:] * deta)
        dN_detadpt  = np.sum(weights[:,np.newaxis]*dN_detadpt_ev,axis=0) 
        dN_err    = np.sqrt( np.sum(np.power(weights[:,np.newaxis]*dN_detadpt_ev_err,2.),axis=0) ) 

        # <pT> integrated over all pT bins
        mean_pt   = np.sum(weights*avg_pt_ev) 
        mean_pt_err = np.sqrt( np.sum(np.power(weights*avg_pt_ev_err,2.)) ) 
        # print(cent, mean_pt, acc_sumpt[cent] , n_ev)

        spectra[cent] = {
            'pt_cents':    pt_cents,
            'dN_deta':          dN_deta,
            'dN_deta_err':      dN_deta_err,
            'dN':          dN_detadpt,
            'dN_err':      dN_err,
            'mean_pt':     float(mean_pt),
            'mean_pt_err': float(mean_pt_err),
            'n_events':    n_ev,
            'rap_window':  rap_window,
        }
    return spectra

def compute_flow_cumulants(masks, records, 
                           irap_mid, irap_subA, irap_subB,
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
    flow_species = {
        'pions': ['pi_minus','pi_plus'],
        'kaons': ['kaon_plus','kaon_minus'],
        'protons': ['proton','anti-proton'],
        'charged_hadrons': 'charged_hadrons'}
    acc={}

    acc['Ref'] = {lab: {
        # standard cumulant: midrapidity
        'avg_pt':     [],   # (n_ev,)
        'c2':     [],   
        'c2_var2':     [],   
        'c4':    [],
        'c4_var2':    [], 
        'c2_4':    [],
        'c2_4_var2':    [],   # (n_ev, n_ord)
        # sub-event A and B
        'c2_sub':   [], 
        'c2_sub_var2':   [],   # (n_ev, n_ord)
        } for lab in labels}
    # now the signal clasws
    for species in flow_species.keys():
        acc[species] = {lab: {
            # pT-differential: signal at midrapidity (coarse grid)
            'avg_pt':     [],   # (n_ev,)
            'd2': [],   # (n_ev, n_pt_flow, n_ord)
            'd2_var2': [],   # (n_ev, n_pt_flow, n_ord)
            'd2_sub': [],   # (n_ev, n_pt_flow, n_ord)
            'd2_sub_var2': [],   # (n_ev, n_pt_flow, n_ord)
        } for lab in labels}

    acc['Gen']={lab: {
        # standard cumulant: midrapidity
        'Mk':     [],   # (n_ev,)
        } for lab in labels}

    # print(f"Computing flow cumulants:")
    # print(f"  Signal species  : {species}  irap_mid={irap_mid}")
    # print(f"  Sub-event A     : {ref_species}  irap_subA={irap_subA}")
    # print(f"  Sub-event B     : {ref_species}  irap_subB={irap_subB}")
    for ifile, fname in enumerate(files):
        if fname == "empty_event":
            fname_ersatz=find_ersatz(files,ifile)
        else: 
            fname_ersatz=fname   

            with h5py.File(fname_ersatz, 'r') as f:
                rap_cuts = f['metadata'].attrs['rap_cuts']
                # print("rap_cuts in FILE:")
                # for i, cut in enumerate(rap_cuts):
                    # print(f"  irap={i}  [{cut[0]:.2f}, {cut[1]:.2f}]")
                # NA = f['particles/charged_hadrons/N_pt'][:, irap_subA, :].sum(axis=-1)
                # NB = f['particles/charged_hadrons/N_pt'][:, irap_subB, :].sum(axis=-1)
                
                # print(f"Sub-event A  mean N = {NA.mean():.2f}  max = {NA.max():.0f}")
                # print(f"Sub-event B  mean N = {NB.mean():.2f}  max = {NB.max():.0f}")

                if pt_cents is None:
                    pt_cents  = f['metadata'].attrs['pt_cents_flow']
                    orders    = list(f['metadata'].attrs['orders'])
                    n_pt      = len(pt_cents)
                    n_ord     = len(orders)

                Qn_mid_pt = {}
                M_mid_pt    ={}

                Qn_A_pt = {}
                M_A_pt    = {}

                Qn_B_pt = {}
                M_B_pt    = {}

                # ── midrapidity signal ─────────────────────────────────────
                # integrated: sum over pT bins -> (n_samp, n_ord)
                prefactor= np.nan
                if fname == "empty_event":
                    prefactor= 0
                else:
                    prefactor= 1

                Qn_mid = 1j* prefactor*f[f'particles/{ref_species}/Qn_imag'] [:, irap_mid, :, :].sum(axis=1)
                Qn_mid += prefactor*f[f'particles/{ref_species}/Qn_real'] [:, irap_mid, :, :].sum(axis=1)
                Q2n_mid = 1j*prefactor*f[f'particles/{ref_species}/Q2n_imag'][:, irap_mid, :, :].sum(axis=1)
                Q2n_mid += prefactor*f[f'particles/{ref_species}/Q2n_real'][:, irap_mid, :, :].sum(axis=1)
                M_mid    = prefactor*f[f'particles/{ref_species}/N_pt_flow'][:, irap_mid, :].sum(axis=1)

                # ── sub-event A ────────────────────────────────────────────
                Qn_subA = 1j*prefactor*f[f'particles/{ref_species}/Qn_imag'][:, irap_subA, :, :].sum(axis=1)
                Qn_subA += prefactor*f[f'particles/{ref_species}/Qn_real'][:, irap_subA, :, :].sum(axis=1)
                M_subA    = prefactor*f[f'particles/{ref_species}/N_pt_flow']   [:, irap_subA, :].sum(axis=1)

                # ── sub-event B ────────────────────────────────────────────
                Qn_subB = 1j*prefactor*f[f'particles/{ref_species}/Qn_imag'][:, irap_subB, :, :].sum(axis=1)
                Qn_subB += prefactor*f[f'particles/{ref_species}/Qn_real'][:, irap_subB, :, :].sum(axis=1)
                M_subB  = prefactor*f[f'particles/{ref_species}/N_pt_flow']   [:, irap_subB, :].sum(axis=1)
                
                for particle in flow_species.keys():
                    if particle=='charged_hadrons':
                        species=flow_species[particle]
                        Qn_mid_pt [particle]= 1j*prefactor*f[f'particles/{species}/Qn_imag'][:, irap_mid, :, :]
                        Qn_mid_pt [particle]+= prefactor*f[f'particles/{species}/Qn_real'][:, irap_mid, :, :]
                        M_mid_pt    [particle]= prefactor*f[f'particles/{species}/N_pt_flow']   [:, irap_mid, :, ]

                        Qn_A_pt [particle]= 1j*prefactor*f[f'particles/{species}/Qn_imag'][:, irap_subA, :, :]
                        Qn_A_pt [particle]+= prefactor*f[f'particles/{species}/Qn_real'][:, irap_subA, :, :]
                        M_A_pt    [particle]= prefactor*f[f'particles/{species}/N_pt_flow']   [:, irap_subA, :, ]

                        Qn_B_pt [particle]= 1j*prefactor*f[f'particles/{species}/Qn_imag'][:, irap_subB, :, :]
                        Qn_B_pt [particle]+= prefactor*f[f'particles/{species}/Qn_real'][:, irap_subB, :, :]
                        M_B_pt    [particle]= prefactor*f[f'particles/{species}/N_pt_flow']   [:, irap_subB, :, ]

                    else:
                        species=flow_species[particle]
                        Qn_mid_pt [particle]= 1j* prefactor*f[f'particles/{species[0]}/Qn_imag'][:, irap_mid, :, :] + f[f'particles/{species[1]}/Qn_imag'][:, irap_mid, :, :]
                        Qn_mid_pt [particle]+= prefactor*f[f'particles/{species[0]}/Qn_real'][:, irap_mid, :, :] + f[f'particles/{species[1]}/Qn_real'][:, irap_mid, :, :]
                        M_mid_pt    [particle]= prefactor*f[f'particles/{species[0]}/N_pt_flow']   [:, irap_mid, :, ] + f[f'particles/{species[1]}/N_pt_flow']   [:, irap_mid, :, ]

                        Qn_A_pt [particle]=  1j* prefactor*f[f'particles/{species[0]}/Qn_imag'][:, irap_subA, :, :] + f[f'particles/{species[1]}/Qn_imag'][:, irap_subA, :, :]
                        Qn_A_pt [particle]+= prefactor*f[f'particles/{species[0]}/Qn_real'][:, irap_subA, :, :] + f[f'particles/{species[1]}/Qn_real'][:, irap_subA, :, :]
                        M_A_pt    [particle]= prefactor*f[f'particles/{species[0]}/N_pt_flow']   [:, irap_subA, :, ] + f[f'particles/{species[1]}/N_pt_flow']   [:, irap_subA, :, ]

                        Qn_B_pt [particle] = 1j* prefactor*f[f'particles/{species[0]}/Qn_imag'][:, irap_subB, :, :] + f[f'particles/{species[1]}/Qn_imag'][:, irap_subB, :, :]
                        Qn_B_pt [particle] += prefactor*f[f'particles/{species[0]}/Qn_real'][:, irap_subB, :, :] + f[f'particles/{species[1]}/Qn_real'][:, irap_subB, :, :]
                        M_B_pt    [particle]= prefactor*f[f'particles/{species[0]}/N_pt_flow']   [:, irap_subB, :, ] + f[f'particles/{species[1]}/N_pt_flow']   [:, irap_subB, :, ]

                        
                n_samp     = M_mid.shape[0]
                file_slice = slice(offset, offset + n_samp)

                for lab in labels:
                    local_mask = masks[lab][file_slice]
                    n_events[lab] += local_mask.sum()
                    if local_mask.sum() == 0:
                        continue

                    M_mid_l    = M_mid[local_mask]
                    Qn_mid_l   = Qn_mid[local_mask] 
                    Q2n_mid_l  = Q2n_mid[local_mask]
                    Qn_subA_l  = Qn_subA[local_mask]
                    M_subA_l   = M_subA[local_mask]
                    Qn_subB_l  = Qn_subB[local_mask] 
                    M_subB_l   = M_subB[local_mask]
                    # print(M_subB_l.shape, Qn_mid_l.shape)

                    c2_mean,c2_var2 = np.zeros(3), np.zeros(3)
                    # c24_mean,c24_var2 = np.zeros(3), np.zeros(3)
                    c2_sub_mean,c2_sub_var2 = np.zeros(3), np.zeros(3)
                    c2_4_mean,c2_4_var2 = np.zeros(3), np.zeros(3)
                    c4_mean,c4_var2 = np.zeros(3), np.zeros(3)
                    # c4_sub_mean,c4_sub_var2 = np.zeros(3), np.zeros(3)
                    len_pt =len(pt_cents)
                    d2_mean, d2_var2={},{}
                    d2_sub_mean, d2_sub_var2 ={},{}
                    for species in flow_species.keys():
                        d2_mean[species], d2_var2[species] =np.zeros((len_pt,3)),np.zeros((len_pt,3))
                        d2_sub_mean[species], d2_sub_var2[species] =np.zeros((len_pt,3)),np.zeros((len_pt,3))

                    M   = M_mid_l
                    MsubA   = M_subA_l
                    MsubB   = M_subB_l

                    Mk  = len(M_mid_l)
                    # print(len(M),Mk, d2_mean)

                    for io, n in enumerate(orders):
                        # N_total_ev[i] = np.mean(acc[cent]['N_tot'][i])
                        # N_total_ev_err[i]   = bootstrap_error_simple(acc[cent]['N_tot'][i])
                        Qm  = Qn_mid_l[:, io]
                        Q2m = Q2n_mid_l[:, io]
                        QA  = Qn_subA_l[:, io]
                        QB  = Qn_subB_l[:, io]
                        # ── standard c2{2} ─────────────────────────────────────
                        # uses only midrapidity particles
                        # self-correlation subtraction: -M removes i==j pairs
                        # denominator M*(M-1) counts all distinct pairs
                        # Ref: Bilandzic PRC 83, 044913 (2011) Eq.(6)
                        ok2 = M >= 2
                        c2_vec = np.zeros_like(M, dtype=float)
                        c2_vec[ok2]  = (np.abs(Qm[ok2])**2 - M[ok2]) / (M[ok2] * (M[ok2] - 1))
                        c2_mean[io] = c2_vec.mean()
                        c2_var2[io] = np.mean(np.power(c2_vec-c2_mean[io],2.))
                        # v2_2[io]     = np.sqrt(max(c2_mean, 0.0))
                        # v2_2_err[io] = _bootstrap_v2(c2_vec, kind='2')

                        # ── sub-event c2{2|AB} ─────────────────────────────────
                        # A from irap_subA, B from irap_subB — different eta windows
                        # so NO self-correlations, denominator is simply M_A * M_B
                        # Ref: Jia PRL 116, 172301 (2016)
                        ok_sub = (MsubA >= 1) & (MsubB >= 1)
                        c2sub_vec = np.zeros_like(M, dtype=float)
                        c2sub_vec[ok_sub] = ((QA[ok_sub] * np.conj(QB[ok_sub])).real / (MsubA[ok_sub] * MsubB[ok_sub]))
                        c2_sub_mean[io] = c2sub_vec.mean()
                        c2_sub_var2[io] = np.mean(np.power(c2sub_vec-c2_sub_mean[io],2.))
                        # v2_2sub[io]     = np.sqrt(max(c2sub_mean, 0.0))
                        # v2_2sub_err[io] = _bootstrap_v2(c2sub_vec, kind='2')

                        ok4 = M >= 4
                        Qm4   = Qm [ok4]
                        Q2m4  = Q2m[ok4]
                        M4    = M  [ok4]
                        
                        #this is c2 but for 4 particles
                        c2_4_vec = np.zeros_like(M, dtype=float)
                        c2_4_vec[ok4]  = (np.abs(Qm4)**2 - M4) / (M4 * (M4 - 1))
                        c2_4_mean[io] = c2_4_vec.mean()
                        c2_4_var2[io] = np.mean(np.power(c2_4_vec-c2_4_mean[io],2.))
                        

                        #Different terms in c4
                        term1    = np.abs(Qm4)**4
                        term2    = np.abs(Q2m4)**2
                        term3    = 2.0 * (Q2m4 * np.conj(Qm4*Qm4)).real
                        term4    = 4.0 * (M4 - 2) * np.abs(Qm4)**2
                        term5    = 2.0 * M4 * (M4 - 3)
                        denom4   = M4 * (M4 - 1) * (M4 - 2) * (M4 - 3)
                        #The c4_vector 
                        c4_vec = np.zeros_like(M, dtype=float)
                        c4_vec[ok4] = (term1 + term2 - term3 - term4 + term5) / denom4
                        
                        c4_mean[io]  = c4_vec.mean() 
                        c4_var2[io]  = np.mean(np.power(c4_vec-c4_mean[io] ,2.))
                         
                    # print (c2_mean,c2_var2)
                    # print (c2_sub_mean,c2_sub_var2)

                        for particle in flow_species.keys():
                            Qn_mid_pt_l=Qn_mid_pt[particle][local_mask]
                            M_mid_pt_l=M_mid_pt[particle][local_mask]
                            # print(Qn_mid_pt_l.shape)

                            Qn_A_pt_l = Qn_A_pt[particle][local_mask]
                            M_A_pt_l = M_A_pt[particle][local_mask]

                            for ib in range(n_pt):
                            ## sub-event differential flow
                                Qb = Qn_A_pt_l[:, ib, io]
                                Mb = M_A_pt_l[:, ib].astype(float)
                                
                                # print(M_subB_l.shape,Mb.shape)
                                ok_b = (Mb >= 1) & (MsubB >= 1)

                                
                                num    = (Qb[ok_b] * np.conj(QB[ok_b])).real
                                den    = Mb[ok_b] * MsubB[ok_b]
                                
                                d2prime_vec = aux.safe_divide( num,den)
                                d2_mean[particle][ib,io]=np.mean(d2prime_vec,axis=0)
                                d2_var2[particle][ib,io]=np.mean(np.power(d2prime_vec-d2_mean[particle][ib,io],2.))
        
                                # vn2_pt_sub[particle][ib, io], vn2_pt_sub_err[particle][ib, io]  = ratio_error(d2prime_avg,d2prime_err, v2_2sub[io],v2_2sub_err[io]) 


                            #  "full" differential flow at midrapidity
                                qn_mid_pt= Qn_mid_pt_l[:, ib, io]
                                mq = M_mid_pt_l[:, ib].astype(float)

                                ok = (mq >= 2) & (M >= 2)
                                num    = (qn_mid_pt[ok] * np.conj(Qm[ok])).real- mq[ok]
                                den    =  mq[ok] * M[ok] - mq[ok]
                                
    

                                d2prime_sub_vec = aux.safe_divide( num,den)

                                d2_sub_mean[particle][ib,io]=np.mean(d2prime_sub_vec,axis=0)
                                d2_sub_var2[particle][ib,io]=np.mean(np.power(d2prime_sub_vec-d2_sub_mean[particle][ib,io],2.))

                    # print(c2_mean,Mk)
                    acc['Gen'][lab]['Mk'].append(Mk)
                    acc['Ref'][lab]['c2'].append(c2_mean)
                    acc['Ref'][lab]['c2_var2'].append(c2_var2)
                    acc['Ref'][lab]['c2_sub'].append(c2_sub_mean)
                    acc['Ref'][lab]['c2_sub_var2'].append(c2_sub_var2)
                    acc['Ref'][lab]['c4'].append(c4_mean)
                    acc['Ref'][lab]['c4_var2'].append(c4_var2)
                    acc['Ref'][lab]['c2_4'].append(c2_mean)
                    acc['Ref'][lab]['c2_4_var2'].append(c2_var2)

                    for species in flow_species.keys():
                        acc[species][lab]['d2'].append(d2_mean[species])
                        acc[species][lab]['d2_var2'].append(d2_var2[species])
                        acc[species][lab]['d2_sub'].append(d2_sub_mean[species])
                        acc[species][lab]['d2_sub_var2'].append(d2_sub_var2[species])

    # now the signal clasws

                offset += n_samp
                if ifile % 50==0:
                    print("Processed ", ifile, " files")

    # ── compute cumulants ──────────────────────────────────────────────
    results = {}

    for lab in labels:
        n_ev = n_events[lab]
        if n_ev == 0:
            print(f"  Warning: no events in {lab}%, skipping.")
            continue
        
        Mka=np.array(acc['Gen'][lab]['Mk'])
        c2a=np.array(acc['Ref'][lab]['c2'])
        c2a_var2=np.array(acc['Ref'][lab]['c2_var2'])
        c2a_sub=np.array(acc['Ref'][lab]['c2_sub'])
        c2a_sub_var2=np.array(acc['Ref'][lab]['c2_sub_var2'])
        c4a=np.array(acc['Ref'][lab]['c4'])
        c4a_var2=np.array(acc['Ref'][lab]['c4_var2'])
        c24a=np.array(acc['Ref'][lab]['c2_4'])
        c24a_var2=np.array(acc['Ref'][lab]['c2_4_var2'])
        
        
        c22, c22_err = cluster_error_obs(c2a,c2a_var2, Mka, len(Mka),n_ev)
        # This is the "naive way to compute this, we can also bootstrap the c22"
        v2_2, v2_2_err = np.sqrt(c22), c22_err/(2*np.sqrt(c22) ) 

        c22_sub, c22_sub_err = cluster_error_obs(c2a_sub,c2a_sub_var2, Mka, len(Mka),n_ev)
        v2_2sub, v2_2sub_err = np.sqrt(c22_sub), c22_sub_err/(2*np.sqrt(c22_sub) ) 

        c4_mean, c4_err = cluster_error_obs(c4a,c4a_var2, Mka, len(Mka),n_ev)
        c24_mean, c24_err = cluster_error_obs(c24a,c24a_var2, Mka, len(Mka),n_ev)

        c4_tot= c4_mean - 2.0 * c24_mean**2
        c4_tot_err= np.sqrt( c4_err*c4_err + np.power(2*c24_mean* c24_err,2)   ) 
        v2_4 = np.zeros(3)
        
        for io in np.arange(3):
            v2_4[io] = np.power(-c4_tot[io],0.25) if c4_tot[io] < 0 else 0.0
        v2_4_err = np.abs(v2_4 * aux.safe_divide(c4_tot_err,c4_tot))

        
        vn2_pt, vn2_pt_err = {},{}
        vn2_pt_sub, vn2_pt_sub_err  = {},{}
        for particle in flow_species.keys():
            # Let's do now differential vn
            d2a=np.array(acc[particle][lab]['d2'])
            d2a_var2=np.array(acc[particle][lab]['d2_var2'])
            d2a_sub=np.array(acc[particle][lab]['d2_sub'])
            d2a_sub_var2=np.array(acc[particle][lab]['d2_sub_var2'])

            d2, d2_err = cluster_error_obs(d2a,d2a_var2, Mka, len(Mka),n_ev)
            d2_sub, d2_sub_err = cluster_error_obs(d2a_sub,d2a_sub_var2, Mka, len(Mka),n_ev)

            # n2_pt_sub[particle][ib, io], vn2_pt_sub_err[particle][ib, io]  = ratio_error(d2prime_avg,d2prime_err, v2_2sub[io],v2_2sub_err[io]) 
            vn2_pt[particle], vn2_pt_err[particle] = np.zeros( d2.shape), np.zeros( d2.shape)
            vn2_pt_sub[particle], vn2_pt_sub_err[particle] = np.zeros( d2.shape), np.zeros( d2.shape)
            for io in np.arange(3):
                for ib in range(n_pt):                                  
                    vn2_pt[particle][ib, io], vn2_pt_err[particle][ib, io] = ratio_error(d2[ib, io], d2_err[ib, io] , v2_2[io],v2_2_err[io]) 
                    vn2_pt_sub[particle][ib, io], vn2_pt_sub_err[particle][ib, io] = ratio_error(d2[ib, io], d2_err[ib, io] , v2_2sub[io],v2_2sub_err[io]) 
            # vn2_pt_sub[particle], vn2_pt_sub_err[particle]  = ratio_error(d2prime_avg,d2prime_err, v2_2sub[io],v2_2sub_err[i

 #c4_vec_mean #- 2.0 * c2_4_mean**2
                          #
                        # v2_4[io]     = (-c4_mean)**0.25 if c4_mean < 0 else np.nan
                        # v2_4_err[io] = _bootstrap_v2(c2_4_vec, kind='4',
                        #                             four_vec=four_vec,
                        #                             c2_mean=c2_4_mean)
        print(v2_2, v2_2_err)
        print(v2_2sub, v2_2sub_err) 
        print(c4_tot, c4_tot_err)
        print(v2_4, v2_4_err)
        d2,d2_err={},{}
        d2_sub,d2_sub_err={},{}

        for particle in flow_species.keys():
            d2=np.array(acc[species][lab]['d2'])
            d2_var2=np.array(acc[species][lab]['d2_var2'])
            d2_sub=np.array(acc[species][lab]['d2_sub'])
            d2_sub_var2=np.array(acc[species][lab]['d2_var2'])


        # print(c2_arra.shape)
        # cluster_error_obs(means, vars2, Mk, K,N)
        
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
            'vn_2_pt_sub':     vn2_pt_sub,      # was 'vn2_pt'
            'vn_2_pt_sub_err': vn2_pt_sub_err,  # was 'vn2_pt_err'
            'n_events':    n_ev,
        }

        # print(f"  {lab:8s}%  n={n_ev:5d}  "
        #       + "  ".join(
        #           f"v{n}{{2}}={v2_2[io]:.4f}  "
        #           f"v{n}{{2|AB}}={v2_2sub[io]:.4f}  "
        #           f"v{n}{{4}}={v2_4[io]:.4f}"
        #           for io, n in enumerate(orders)))

    return results


def cluster_error_obs(means, vars2, Mk, K,N):

    # Reshape Mk for broadcasting with means/vars2 of any shape
    # (K,) -> (K, 1, 1, ..., 1)
    Mk_bcast = Mk.reshape((K,) + (1,) * (means.ndim - 1))
    # Compute grand mean (sum over batch axis)
    GrandMean = np.sum(Mk_bcast * means, axis=0) / float(N)
    # Compute within-batch variance
    Sw2 = np.sum(Mk_bcast * vars2, axis=0) / float(N - K)
    # Compute between-batch variance
    SB2 = np.sum(Mk_bcast * np.power(means - GrandMean, 2.), axis=0) / float(K - 1.0)
    
    # M0 and ICC (scalars, since Mk is 1D)
    M0 = (N - np.sum(Mk * Mk) / float(N)) / (K - 1.)
    ICC = (SB2 - Sw2) / (SB2 + (M0 - 1.) * Sw2)
    
    # SE (broadcasts to shape of Sw2)
    SEXBar = np.sqrt((Sw2 / N) * (1 + (M0 - 1) * ICC))
        # Sw2, SB2, ICC,
    return GrandMean, SEXBar

def _bootstrap_v2(c2_vec, kind='2', n_boot=NBoot , four_vec=None, c2_mean=None):
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


def _bootstrap_rat(rat, ref_norm, n_boot=NBoot ):
    """Bootstrap error on pT-differential vn{2}."""
    n    = len(rat)
    boot = np.array([
        rat[np.random.randint(0, n, n)].mean() / ref_norm
        for _ in range(n_boot)
    ])
    return np.nanstd(boot)

def bootstrap_error(data, statistic=np.mean, n_boot=NBoot ,axis=0):
    """
    Compute bootstrap error for 2D array, reducing along specified axis.
    
    Parameters
    ----------
    data : np.ndarray, shape (N, M)
        Input 2D array
    axis : int
        Axis to compute statistic along (0 or 1)
        - axis=0: resample rows, compute along columns
        - axis=1: resample columns, compute along rows
    statistic : callable
        Function to compute (default: np.mean)
    n_boot : int
        Number of bootstrap samples
    
    Returns
    -------
    error : np.ndarray
        Error for each element after reduction
    
    Examples
    --------
    >>> data.shape = (1000, 50)  # 1000 events, 50 pT bins
    >>> err = bootstrap_error_axis(data, axis=0)  # Error per pT bin
    >>> err.shape = (50,)
    """
    n = data.shape[axis]  # Size of axis to resample
    
    # Determine output shape
    if axis == 0:
        out_shape = data.shape[1]
    else:
        out_shape = data.shape[0]
    
    boot_samples = np.zeros((n_boot, out_shape))
    
    for ib in range(n_boot):
        # Resample along the specified axis
        idx = np.random.randint(0, n, n)
        
        if axis == 0:
            resampled = data[idx, :]
        else:
            resampled = data[:, idx]
        
        # Compute statistic along that axis
        boot_samples[ib] = statistic(resampled, axis=axis)
    
    # Error = std across bootstrap samples
    return np.nanstd(boot_samples, axis=0)


def bootstrap_error_simple(data, statistic=np.mean, n_boot=NBoot ):
    """
    Compute bootstrap error on any statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    statistic : callable
        Function to compute statistic (default: np.mean)
        Examples: np.mean, np.median, np.std
    n_boot : int
        Number of bootstrap samples (default: 200)
    
    Returns
    -------
    error : float
        Standard deviation of bootstrap samples
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> err = bootstrap_error(data)  # Error on mean
    >>> err_median = bootstrap_error(data, np.median)
    >>> err_std = bootstrap_error(data, np.std)
    """
    n = len(data)
    boot_samples = np.zeros(n_boot)
    
    for ib in range(n_boot):
        # Resample with replacement
        idx = np.random.randint(0, n, n)
        boot_samples[ib] = statistic(data[idx])
    
    # Error = std dev of bootstrap samples
    return np.nanstd(boot_samples)