
def compute_integrated_flow(data, mask, meta, species='pi_plus', irap=1):
    """
    Compute integrated (reference) flow vn{2} and vn{4} using the full
    integrated Q-vector summed over all pT bins.

    This is the standard 'reference flow' used in heavy-ion analyses —
    it uses all particles in the rapidity window without pT binning,
    giving maximum statistical weight.

    The integrated Q-vector is:
        Qn_int = sum_{pT bins} Qn(pT) = sum_{all particles} e^{i n phi}
    which is exact because the Q-vectors are additive over disjoint
    particle subsets.

    Parameters
    ----------
    data    : dict — output of load_keys(), must contain
              N_pt, Qn_real, Qn_imag, Q2n_real, Q2n_imag
    mask    : boolean np.ndarray — centrality selection
    meta    : dict — from extract_dNch()
    species : str  — which species
    irap    : int  — rapidity cut index

    Returns
    -------
    results : dict {n: {'vn2', 'vn4', 'vn2_err', 'vn4_err', 'c2', 'c4'}}
    """
    orders = meta['orders']
    d      = data[species]

    # Apply centrality mask.
    N_pt     = d['N_pt']    [mask, irap, :]      # (n_sel, n_pt)
    Qn_real  = d['Qn_real'] [mask, irap, :, :]   # (n_sel, n_pt, n_ord)
    Qn_imag  = d['Qn_imag'] [mask, irap, :, :]
    Q2n_real = d['Q2n_real'][mask, irap, :, :]
    Q2n_imag = d['Q2n_imag'][mask, irap, :, :]

    # Integrated quantities: sum over pT axis (-2).
    # This is exact — Q-vectors are additive over particle subsets.
    M_int   = N_pt.sum(axis=-1)                              # (n_sel,)
    Qn_int  = (Qn_real.sum(axis=-2)
               + 1j * Qn_imag.sum(axis=-2))                 # (n_sel, n_ord)
    Q2n_int = (Q2n_real.sum(axis=-2)
               + 1j * Q2n_imag.sum(axis=-2))                # (n_sel, n_ord)

    # Four-particle cumulant requires M >= 4.
    ok = M_int >= 4
    M   = M_int[ok].astype(float)
    print(f"  {species} |y|<{meta['rap_cuts'][irap][1]:.1f}: "
          f"{ok.sum()} / {len(mask.nonzero()[0])} events pass M>=4 cut")

    results = {}
    for io, n in enumerate(orders):
        Qn  = Qn_int [ok, io]
        Q2n = Q2n_int[ok, io]

        # ── c2{2}: two-particle cumulant ───────────────────────────────
        c2_vec  = (np.abs(Qn)**2 - M) / (M * (M - 1))
        c2_mean = c2_vec.mean()

        # ── c2{4}: four-particle cumulant ──────────────────────────────
        term1    = np.abs(Qn)**4
        term2    = np.abs(Q2n)**2
        term3    = 2.0 * (Q2n * np.conj(Qn)**2).real
        term4    = 4.0 * (M - 2) * np.abs(Qn)**2
        term5    = 2.0 * M * (M - 3) * (M - 1)
        denom    = M * (M - 1) * (M - 2) * (M - 3)
        four_vec = (term1 + term2 - term3 - term4 + term5) / denom
        c4_mean  = four_vec.mean() - 2.0 * c2_mean**2

        vn2 = np.sqrt(max(c2_mean, 0.0))
        vn4 = (-c4_mean)**0.25 if c4_mean < 0 else float('nan')

        vn2_err, vn4_err = _bootstrap_errors(c2_vec, four_vec, n_boot=500)

        results[n] = {
            'vn2':     vn2,     'vn2_err': vn2_err,
            'vn4':     vn4,     'vn4_err': vn4_err,
            'c2':      c2_mean, 'c4':      c4_mean,
            'n_events': int(ok.sum()),
        }

        print(f"  v{n}{{2}} = {vn2:.5f} +/- {vn2_err:.5f}   "
              f"v{n}{{4}} = {vn4:.5f} +/- {vn4_err:.5f}")

    return results


def compute_ptdiff_flow(data, mask, meta, species='pi_plus', irap=1,
                        ref_species=None, irap_ref=None):
    """
    Compute pT-differential flow vn{2}(pT) using the scalar product method.

    vn{2}(pT) is computed as:

        vn{2}(pT) = c2(pT) / sqrt(c2_ref)

    where:
        c2(pT)  = <Re(Qn(pT) * Qn_ref*)> / (M(pT) * M_ref)
                  with self-correlations subtracted when ref == signal
        c2_ref  = integrated two-particle cumulant from the reference

    Using a separate reference species (e.g. charged particles for the
    reference, pions for the signal) eliminates self-correlations entirely
    and is the cleanest approach.

    Parameters
    ----------
    data        : dict — must contain N_pt, Qn_real, Qn_imag for both
                  species and Q2n arrays for the reference
    mask        : boolean np.ndarray — centrality selection
    meta        : dict
    species     : str  — signal species for pT-differential measurement
    irap        : int  — rapidity cut index for signal
    ref_species : str or None — reference species for normalisation.
                  If None, uses the same species as signal (self-correlation
                  subtraction applied automatically).
    irap_ref    : int or None — rapidity cut for reference.
                  If None, uses same as irap.

    Returns
    -------
    results : dict {n: {'vn2_pt': np.ndarray shape (n_pt,),
                        'vn2_pt_err': np.ndarray,
                        'pt_cents': np.ndarray}}
    """
    if ref_species is None:
        ref_species = species
    if irap_ref is None:
        irap_ref = irap

    same = (ref_species == species) and (irap_ref == irap)

    orders   = meta['orders']
    pt_cents = meta['pt_cents']
    n_pt     = len(pt_cents)
    d_sig    = data[species]
    d_ref    = data[ref_species]

    # Apply centrality mask.
    # Signal arrays — pT-differential, shape (n_sel, n_pt, n_ord)
    N_pt_sig  = d_sig['N_pt']   [mask, irap, :]
    Qn_sig_r  = d_sig['Qn_real'][mask, irap, :, :]
    Qn_sig_i  = d_sig['Qn_imag'][mask, irap, :, :]

    # Reference arrays — integrated, shape (n_sel, n_ord)
    N_ref    = d_ref['N_pt']    [mask, irap_ref, :].sum(axis=-1)
    Qn_ref   = (d_ref['Qn_real'][mask, irap_ref, :, :].sum(axis=-2)
                + 1j * d_ref['Qn_imag'][mask, irap_ref, :, :].sum(axis=-2))
    Q2n_ref  = (d_ref['Q2n_real'][mask, irap_ref, :, :].sum(axis=-2)
                + 1j * d_ref['Q2n_imag'][mask, irap_ref, :, :].sum(axis=-2))

    M_ref = N_ref.astype(float)
    ok_ref = M_ref >= 4

    results = {}
    for io, n in enumerate(orders):
        vn2_pt     = np.full(n_pt, np.nan)
        vn2_pt_err = np.full(n_pt, np.nan)

        # ── reference: integrated c2 for normalisation ─────────────────
        Qn_r  = Qn_ref [ok_ref, io]
        Q2n_r = Q2n_ref[ok_ref, io]
        M_r   = M_ref  [ok_ref]

        c2_ref_vec = (np.abs(Qn_r)**2 - M_r) / (M_r * (M_r - 1))
        c2_ref     = c2_ref_vec.mean()

        if c2_ref <= 0:
            print(f"  Warning: c2_ref <= 0 for n={n}, cannot normalise vn2(pT)")
            results[n] = {'vn2_pt': vn2_pt, 'vn2_pt_err': vn2_pt_err,
                          'pt_cents': pt_cents}
            continue

        ref_norm = np.sqrt(c2_ref)

        # ── signal: per pT bin ─────────────────────────────────────────
        for ib in range(n_pt):
            M_b  = N_pt_sig[:, ib].astype(float)
            Qn_b = (Qn_sig_r[:, ib, io] + 1j * Qn_sig_i[:, ib, io])

            # Need at least some particles in this bin AND reference ok.
            ok_b = ok_ref & (M_b >= 1)
            if ok_b.sum() < 10:
                continue

            M_b_ok  = M_b  [ok_b]
            Qn_b_ok = Qn_b [ok_b]
            Qn_r_ok = Qn_ref[ok_b, io]
            M_r_ok  = M_ref[ok_b]

            if same:
                # Self-correlation subtraction: the bin particles are a
                # subset of the reference, so we subtract their contribution.
                # Numerator: Re(Qn_b * Qn_ref*) - M_b (self pairs)
                # Denominator: M_b * (M_ref - 1)
                num = (Qn_b_ok * np.conj(Qn_r_ok)).real - M_b_ok
                den = M_b_ok * (M_r_ok - 1)
            else:
                # Different species: no self-correlations to subtract.
                num = (Qn_b_ok * np.conj(Qn_r_ok)).real
                den = M_b_ok * M_r_ok

            ok_den   = den > 0
            c2_pt    = np.mean(num[ok_den] / den[ok_den])
            vn2_pt[ib] = c2_pt / ref_norm

            # Bootstrap error for this pT bin.
            n_ok = ok_den.sum()
            rat  = (num[ok_den] / den[ok_den])
            boot = np.array([
                np.mean(rat[np.random.randint(0, n_ok, n_ok)]) / ref_norm
                for _ in range(200)
            ])
            vn2_pt_err[ib] = np.std(boot)

        results[n] = {
            'vn2_pt':     vn2_pt,
            'vn2_pt_err': vn2_pt_err,
            'pt_cents':   pt_cents,
            'c2_ref':     c2_ref,
        }

    return results


def compute_flow_cumulants(data, mask, meta, species='pi_plus', irap=1):
    """
    Compute vn{2} and vn{4} for a given centrality class.

    Parameters
    ----------
    data    : dict — output of load_keys(), must contain
              N_pt, Qn_real, Qn_imag, Q2n_real, Q2n_imag
    mask    : boolean np.ndarray shape (n_total_events,) — centrality selection
    meta    : dict — from extract_dNch(), contains orders, rap_cuts, pt_cents
    species : str  — which species to compute flow for
    irap    : int  — rapidity cut index

    Returns
    -------
    results : dict {n: {'vn2': float, 'vn4': float, 'c2': float, 'c4': float}}
    """
    orders  = meta['orders']
    d       = data[species]

    # Apply centrality mask — all subsequent operations are on selected events only.
    # Shape after masking: (n_selected, n_rap_cuts, n_pt, n_orders)
    N_pt     = d['N_pt']    [mask, irap, :].astype(float)   # (n_sel, n_pt)
    Qn_real  = d['Qn_real'] [mask, irap, :, :]              # (n_sel, n_pt, n_ord)
    Qn_imag  = d['Qn_imag'] [mask, irap, :, :]
    Q2n_real = d['Q2n_real'][mask, irap, :, :]
    Q2n_imag = d['Q2n_imag'][mask, irap, :, :]

    # Integrated Q-vectors: sum over pT bins -> shape (n_sel, n_ord)
    M_int   = N_pt.sum(axis=-1)                              # (n_sel,)
    Qn_int  = (Qn_real.sum(axis=-2)
               + 1j * Qn_imag.sum(axis=-2))                 # (n_sel, n_ord)
    Q2n_int = (Q2n_real.sum(axis=-2)
               + 1j * Q2n_imag.sum(axis=-2))                # (n_sel, n_ord)

    # Only keep events with enough particles for 4-particle cumulant.
    ok = M_int >= 4                                          # (n_sel,) boolean

    results = {}

    for io, n in enumerate(orders):
        Qn  = Qn_int [ok, io]   # shape (n_ok,)
        Q2n = Q2n_int[ok, io]
        M   = M_int  [ok]

        # ── two-particle cumulant c2{2} ────────────────────────────────
        # c2 = (<|Qn|^2> - M) / (M(M-1))
        # The -M term removes the trivial self-correlation (i=j pairs).
        # Reference: Bilandzic et al., PRC 83, 044913 (2011), Eq. (6)
        c2_vec  = (np.abs(Qn)**2 - M) / (M * (M - 1))      # (n_ok,)
        c2_mean = c2_vec.mean()

        # ── four-particle cumulant c2{4} ───────────────────────────────
        # Built from the generic 4-particle Q-vector formula.
        # Reference: Bilandzic et al., PRC 83, 044913 (2011), Eq. (13)
        #
        # <4> = (|Qn|^4 + |Q2n|^2 - 2*Re(Q2n * Qn*^2)
        #        - 4*(M-2)*|Qn|^2 + 2*M*(M-3)) / (M*(M-1)*(M-2)*(M-3))
        #
        # c2{4} = <4> - 2*<2>^2
        term1   = np.abs(Qn)**4
        term2   = np.abs(Q2n)**2
        term3   = 2.0 * (Q2n * np.conj(Qn)**2).real
        term4   = 4.0 * (M - 2) * np.abs(Qn)**2
        term5   = 2.0 * M * (M - 3) * (M - 1)
        denom   = M * (M - 1) * (M - 2) * (M - 3)

        four_vec = (term1 + term2 - term3 - term4 + term5) / denom  # <4> per event
        c4_mean  = four_vec.mean() - 2.0 * c2_mean**2

        # ── vn{2} and vn{4} ────────────────────────────────────────────
        vn2 = np.sqrt(max(c2_mean, 0.0))
        vn4 = (-c4_mean)**0.25 if c4_mean < 0 else float('nan')

        # ── statistical uncertainties via bootstrap ────────────────────
        vn2_err, vn4_err = _bootstrap_errors(c2_vec, four_vec, n_boot=200)

        results[n] = {
            'vn2':    vn2,
            'vn4':    vn4,
            'vn2_err': vn2_err,
            'vn4_err': vn4_err,
            'c2':     c2_mean,
            'c4':     c4_mean,
            'n_events': int(ok.sum()),
        }

        print(f"  v{n}{{2}} = {vn2:.5f} +/- {vn2_err:.5f}   "
              f"v{n}{{4}} = {vn4:.5f} +/- {vn4_err:.5f}   "
              f"(N={ok.sum()})")

    return results

def load_all_and_select_centrality(filenames, bins_percentile,
                                   species_to_load=None,
                                   keys_to_load=None):
    """
    Single-pass loader with memory controls.

    Parameters
    ----------
    filenames        : list of str
    bins_percentile  : list of percentile edges
    species_to_load  : list of str or None — if None, load all species.
                       Pass e.g. ['charged', 'pi_plus'] to save memory.
    keys_to_load     : set of str or None — if None, load all keys.
                       Pass e.g. {'N_pt', 'Qn_real', 'Qn_imag', 'Q2n_real', 'Q2n_imag'}
                       to skip pQn arrays until needed.
    """
    filenames = sorted(filenames)

    all_arrays = {}
    all_dNch   = []
    meta       = None

    print(f"Loading {len(filenames)} files ...")

    for fname in filenames:
        with h5py.File(fname, 'r') as f:
            m        = f['metadata']
            rap_cuts = m.attrs['rap_cuts']

            if meta is None:
                meta = {
                    'pt_bins':     m.attrs['pt_bins'],
                    'pt_cents':    m.attrs['pt_cents'],
                    'orders':      list(m.attrs['orders']),
                    'rap_cuts':    {i: list(rap_cuts[i])
                                   for i in range(len(rap_cuts))},
                    'species_pdg': {sp: m['species_pdg'].attrs[sp]
                                   for sp in m['species_pdg'].attrs},
                }

            # centrality estimator — always load this regardless of species_to_load
            irap_cent = _find_rap_cut_index(rap_cuts, target=[0.0, 0.5])
            rap_width = rap_cuts[irap_cent, 1] - rap_cuts[irap_cent, 0]
            N_ch      = f['particles/charged/N_pt'][:, irap_cent, :].sum(axis=-1)
            all_dNch.append(N_ch / (2.0 * rap_width))

            # load particle arrays with optional filtering
            species = species_to_load or list(f['particles'].keys())
            for name in species:
                if name not in f['particles']:
                    continue
                grp  = f[f'particles/{name}']
                keys = keys_to_load or set(grp.keys())
                if name not in all_arrays:
                    all_arrays[name] = {key: [] for key in keys}
                for key in keys:
                    if key in grp:
                        all_arrays[name][key].append(grp[key][:])

    # concatenate
    print("Concatenating ...")
    dNch_all   = np.concatenate(all_dNch)
    merged_all = {
        name: {key: np.concatenate(arrs, axis=0)
               for key, arrs in data.items()}
        for name, data in all_arrays.items()
    }

    # estimate memory usage
    total_bytes = sum(
        arr.nbytes
        for data in merged_all.values()
        for arr in data.values()
    )
    print(f"Total sampling events : {len(dNch_all)}")
    print(f"Memory usage          : {total_bytes / 1e9:.2f} GB")

    # centrality selection in memory
    edges     = np.percentile(dNch_all, 100 - np.array(bins_percentile))
    cent_data = {}

    for i in range(len(bins_percentile) - 1):
        label    = f'{bins_percentile[i]}-{bins_percentile[i+1]}'
        mult_min = edges[i + 1]
        mult_max = edges[i]
        mask     = (dNch_all >= mult_min) & (dNch_all <= mult_max)
        n_sel    = mask.sum()

        print(f"  Centrality {label:6s}%  "
              f"dNch/deta in [{mult_min:.1f}, {mult_max:.1f}]  "
              f"N = {n_sel}")

        merged_cent = {
            name: {key: arr[mask] for key, arr in data.items()}
            for name, data in merged_all.items()
        }
        meta_cent = dict(meta)
        meta_cent['centrality_label']  = label
        meta_cent['dNch_range']        = [mult_min, mult_max]
        meta_cent['n_sampling_events'] = int(n_sel)
        cent_data[label] = (merged_cent, meta_cent)

    return cent_data

def extract_dNch(filenames):
    """
    Minimal first pass: load only N_pt for charged particles to build
    the centrality estimator. Opens each file once, reads one dataset.

    Returns
    -------
    dNch_all : np.ndarray shape (n_total_sampling_events,)
    meta     : dict — pt_bins, orders, rap_cuts etc. from first file
    """
    filenames = sorted(filenames)
    all_dNch  = []
    meta      = None

    for fname in filenames:
        with h5py.File(fname, 'r') as f:
            m        = f['metadata']
            rap_cuts = m.attrs['rap_cuts']

            if meta is None:
                meta = {
                    'pt_bins':     m.attrs['pt_bins'],
                    'pt_cents':    m.attrs['pt_cents'],
                    'orders':      list(m.attrs['orders']),
                    'rap_cuts':    {i: list(rap_cuts[i])
                                   for i in range(len(rap_cuts))},
                    'species_pdg': {sp: m['species_pdg'].attrs[sp]
                                   for sp in m['species_pdg'].attrs},
                }

            irap_cent = _find_rap_cut_index(rap_cuts, target=[0.0, 0.5])
            rap_width = rap_cuts[irap_cent, 1] - rap_cuts[irap_cent, 0]

            # This is the only dataset read in this pass.
            N_ch = f['particles/charged/N_pt'][:, irap_cent, :].sum(axis=-1)
            all_dNch.append(N_ch / (2.0 * rap_width))

    return np.concatenate(all_dNch), meta


def make_centrality_masks(dNch_all, bins_percentile):
    """
    Compute boolean centrality masks from the dNch distribution.
    These masks are tiny (one bool per sampling event) and can be kept
    in memory for the entire analysis session.

    Returns
    -------
    masks : dict {label: np.ndarray of bool, shape (n_total_events,)}
    edges : np.ndarray — multiplicity edges corresponding to percentile cuts
    """
    edges  = np.percentile(dNch_all, 100 - np.array(bins_percentile))
    masks  = {}

    for i in range(len(bins_percentile) - 1):
        label    = f'{bins_percentile[i]}-{bins_percentile[i+1]}'
        mult_min = edges[i + 1]
        mult_max = edges[i]
        masks[label] = (dNch_all >= mult_min) & (dNch_all <= mult_max)

        print(f"  Centrality {label:6s}%  "
              f"dNch/deta in [{mult_min:.1f}, {mult_max:.1f}]  "
              f"N = {masks[label].sum()}")

    return masks, edges


def load_keys(filenames, species, keys):
    """
    Load only specific species and array keys across all files.
    Each file is opened once. Returns concatenated arrays.

    Parameters
    ----------
    filenames : list of str
    species   : list of str — e.g. ['pi_plus', 'charged']
    keys      : set of str  — e.g. {'N_pt', 'Qn_real', 'Qn_imag'}

    Returns
    -------
    data : dict {species_name: {key: np.ndarray}}
           Arrays have shape (n_total_sampling_events, ...)
    """
    filenames = sorted(filenames)
    accum     = {name: {key: [] for key in keys} for name in species}

    for fname in filenames:
        with h5py.File(fname, 'r') as f:
            for name in species:
                grp = f[f'particles/{name}']
                for key in keys:
                    accum[name][key].append(grp[key][:])

    # estimate memory
    total_bytes = sum(
        arr.nbytes
        for arrays in accum.values()
        for arr_list in arrays.values()
        for arr in arr_list
    )
    print(f"Loaded {', '.join(species)} "
          f"keys={keys} — {total_bytes/1e9:.2f} GB")

    return {
        name: {key: np.concatenate(arrs, axis=0)
               for key, arrs in arrays.items()}
        for name, arrays in accum.items()
    }

def _bootstrap_errors(c2_vec, four_vec, n_boot=200):
    """
    Estimate statistical uncertainties on vn{2} and vn{4} via bootstrap
    resampling of the per-event cumulant vectors.

    Bootstrap is preferred over error propagation here because vn{4}
    involves a fractional power of c4 which makes analytic error
    propagation messy and unreliable when c4 is close to zero.

    Parameters
    ----------
    c2_vec   : np.ndarray (n_events,) — per-event <2> values
    four_vec : np.ndarray (n_events,) — per-event <4> values
    n_boot   : int — number of bootstrap samples

    Returns
    -------
    vn2_err, vn4_err : float
    """
    n      = len(c2_vec)
    vn2_bs = np.zeros(n_boot)
    vn4_bs = np.zeros(n_boot)

    for ib in range(n_boot):
        idx      = np.random.randint(0, n, size=n)   # resample with replacement
        c2_b     = c2_vec[idx].mean()
        c4_b     = four_vec[idx].mean() - 2.0 * c2_b**2
        vn2_bs[ib] = np.sqrt(max(c2_b,  0.0))
        vn4_bs[ib] = (-c4_b)**0.25 if c4_b < 0 else np.nan

    vn2_err = np.std(vn2_bs)
    vn4_err = np.nanstd(vn4_bs)

    return vn2_err, vn4_err

# ── step 1: compute centrality mask ONCE ──────────────────────────────
# Only loads N_pt for charged particles — very cheap.
dNch_all, meta = pr.extract_dNch(files)

bins_percentile = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
masks, edges    = pr.make_centrality_masks(dNch_all, bins_percentile)
# masks is a dict {'0-5': boolean array, '5-10': boolean array, ...}

# ── step 2: load and analyse observable by observable ─────────────────

# spectra + <pT>
data = pr.load_keys(files, species=['pi_plus', 'kaon_plus', 'proton'],
                    keys={'N_pt', 'sum_pt'})
for label, mask in masks.items():
    compute_spectra(data, mask, meta)

# vn{2}, vn{4}
data = pr.load_keys(files, species=['pi_plus', 'charged'],
                    keys={'N_pt', 'Qn_real', 'Qn_imag', 'Q2n_real', 'Q2n_imag'})
for label, mask in masks.items():
    compute_flow(data, mask, meta)

# <vn * <pT>> correlator
data = pr.load_keys(files, species=['pi_plus'],
                    keys={'N_pt', 'sum_pt', 'Qn_real', 'Qn_imag',
                          'pQn_real', 'pQn_imag'})
for label, mask in masks.items():
    compute_vn_pt_correlator(data, mask, meta)

