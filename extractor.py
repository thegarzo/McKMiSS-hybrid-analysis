
"""
Copyright (c) 2026, Oscar Garcia-Montero
For private use only. All rights reserved.

====================
Per-node analysis script for heavy-ion collision simulations running on a
Condor/grid cluster. Each node runs a single MUSIC hydrodynamic event with
multiple oversampled iSS particlisation events.

Pipeline assumed:
-> MUSIC 2+1D boost-invariant hydrodynamics (from arbitrary early stages
-> iSS (Cooper-Frye particlisation, binary output)
-> this script

If using SMASH, the reader needs to be adapted to read the SMASH output format instead of iSS's binary format.

What this script does
---------------------
1. Reads the iSS binary particle output (particle_samples.bin).
2. Computes per-sampling-event Q-vector summaries in pT bins for each
   particle species. These summaries are the minimal objects needed to
   reconstruct all desired observables (spectra, <pT>, vn{2}, vn{4},
   radial flow) during the final merged analysis.
3. Stores everything — particle summaries + MUSIC config + MCDipper config
   — in a self-documenting HDF5 file.

Why Q-vectors in pT bins?
--------------------------
Storing full particle lists off the node is expensive (hundreds of MB per
event). Instead we reduce each sampling event to a small set of per-pT-bin
Q-vectors (~10 MB total per node across all species). These Q-vectors are
sufficient to reconstruct:
    - dN/dydpT spectra            (from N_pt, sum_pt)
    - integrated <pT>             (sum over N_pt, sum_pt)
    - pT-differential vn{2}(pT)  (from Qn_pt and integrated Qn)
    - integrated vn{2}, vn{4}    (sum Qn_pt over pT bins)
    - <v2 * <pT>> correlator     (from pQn_pt: pT-weighted Q-vector)

Usage
-----
    python per_node_analysis.py <event_id> <binary_file> \
                                <music_config> <mcdipper_config> <output_dir>

Example:
    python per_node_analysis.py 42 particle_samples.bin \
        music_input mcdipper_config.yaml results/
"""

import numpy as np 
import sys
import os

import parser as pa
import analyser as an   

def main():
    """
    Main entry point for per-node execution.

    Command-line arguments (all positional):
        1. hydro_event_id       : int — unique ID for this hydro event
        2. binary_file          : str — path to particle_samples.bin
        3. output_dir           : str — directory to write HDF5 output

    Output file will be named:
        <output_dir>/hydro_event_<id:04d>.h5
    """
    # ── parse command-line arguments ───────────────────────────────────────
    if len(sys.argv) < 4:
        print("Usage: python per_node_analysis.py "
              "<event_id> <binary_file> <output_dir>")
        sys.exit(1)

    event_id             = int(sys.argv[1])
    binary_file          = sys.argv[2]
    output_dir           = sys.argv[3]


    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'hydro_event_{event_id:04d}.h5')

    print(f"[Event {event_id}] Reading {binary_file} ...")

    # ── read particle data ─────────────────────────────────────────────────
    events = pa.read_iSS_binary(binary_file)
    print(f"[Event {event_id}] Loaded {len(events)} sampling events")

    # Run sanity check before committing to the full computation.
    an.sanity_check(events)

    # ── compute Q-vector summaries ─────────────────────────────────────────
    print(f"[Event {event_id}] Computing Q-vector summaries ...")
    summary = an.compute_summary(events)

    # Print a quick multiplicity check per species.
    for name, data in summary.items():
        mean_N = data['N_pt'].sum(axis=1).mean()
        print(f"  {name:12s}  <N> per sampling event = {mean_N:.1f}")

    # ── save to HDF5 ───────────────────────────────────────────────────────
    print(f"[Event {event_id}] Writing {output_file} ...")
    an.save_hdf5(
        filename             = output_file,
        summary              = summary,
        hydro_event_id       = event_id
    )

    print(f"[Event {event_id}] Done. Output: {output_file}")
    
if __name__ == "__main__":
    main()
