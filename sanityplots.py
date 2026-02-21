import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

import glob
from parameters import *
import processer as pr

import matplotlib.cm as cm



###################################################### EVENTS ###########################################################

def plot_events_selected(infos, records):
    # quick look at the centrality bin definitions
    fig, ax = plt.subplots()
    ax.hist(records['dNch_deta'], bins=80, color='steelblue', alpha=0.7)

    for label, d in infos.items():
        # print(label)
        ax.axvline(d['dNch_range'][0], color='red', lw=0.8, ls='--')
        ax.text(d['dNch_range'][0]*1.008, ax.get_ylim()[1]*0.9,
                label, fontsize=7, rotation=90, color='red', ha='right')

    ax.set_xlabel(r'$dN_{ch}/d\eta$')
    ax.set_ylabel('Counts')
    ax.set_title(f"Centrality definition  "
                f"|y| in ({records['rap_window'][0]:.1f}, "
                f"{records['rap_window'][1]:.1f})")
    plt.tight_layout()
    plt.savefig('SanityPlots/centrality_definition.pdf')
    print("Saved centrality_definition.pdf")

###################################################### dNchdeta ###########################################################

def plot_dNch_deta(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = cm.plasma(np.linspace(0.1, 0.9, len(results)))

    for (lab, res), color in zip(results.items(), colors):
        ax.plot(res['eta_cents'], res['dNch_deta'],
                '-', color=color, label=f'{lab}%')

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$dN_{ch}/d\eta$')
    ax.set_title('Charged hadron pseudorapidity distribution')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('SanityPlots/dNch_deta.pdf')
    print("Saved SanityPlots/dNch_deta.pdf")

###################################################### PT ###########################################################
def plot_pt_spectra(spectra, species):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = cm.plasma(np.linspace(0.1, 0.9, len(spectra)))

    for (label, sp), color in zip(spectra[species].items(), colors):
        pt  = sp['pt_cents']
        dN  = sp['dN']
        err = sp['dN_err']
        ok  = dN > 0

        axes[0].errorbar(pt[ok], dN[ok], yerr=err[ok],
                        fmt='o-', ms=3, color=color,
                        label=f'{label}%  <pT>={sp["mean_pt"]:.3f} GeV')

        # also plot scaled by <pT> for shape comparison
        axes[1].errorbar(pt[ok], dN[ok] / dN[ok].max(), yerr=err[ok] / dN[ok].max(),
                        fmt='o-', ms=3, color=color, label=f'{label}%')

    axes[0].set_yscale('log')
    # axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$p_T$ (GeV)')
    axes[0].set_ylabel(r'$\frac{1}{2\pi p_T}\frac{dN}{dy\,dp_T}$ (GeV$^{-2}$)')
    axes[0].set_title(r'$\pi^+$ spectra by centrality')
    axes[0].legend(fontsize=7)

    axes[1].set_yscale('log')
    # axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$p_T$ (GeV)')
    axes[1].set_ylabel('Normalised yield')
    axes[1].set_title(r'$\pi^+$ shape comparison')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('SanityPlots/spectra_'+species+ '.pdf')
    print("Saved spectra_"+species+".pdf")

###################################################### FLOW ###########################################################
def plot_flows(flow):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = cm.plasma(np.linspace(0.1, 0.9, len(flow)))

    cent_labels = list(flow.keys())
    cent_x      = np.arange(len(cent_labels))

    for io, n in enumerate(flow[cent_labels[0]]['orders']):
        # standard v2{2}
        vn2  = np.array([flow[lab]['vn_2'][io]    for lab in cent_labels])
        err  = np.array([flow[lab]['vn_2_err'][io] for lab in cent_labels])
        axes[0].errorbar(cent_x, vn2, yerr=err, fmt='o-', ms=5,
                         label=f'v{n}{{2}}')

        # sub-event v2{2|AB} — skip if all nan
        vn2sub = np.array([flow[lab]['vn_2sub'][io] for lab in cent_labels])
        if not np.all(np.isnan(vn2sub)):
            err_sub = np.array([flow[lab]['vn_2sub_err'][io] for lab in cent_labels])
            axes[0].errorbar(cent_x, vn2sub, yerr=err_sub, fmt='s--', ms=5,
                             label=f'v{n}{{2|AB}}')

        # v2{4} — skip if all nan
        vn4 = np.array([flow[lab]['vn_4'][io] for lab in cent_labels])
        if not np.all(np.isnan(vn4)):
            err4 = np.array([flow[lab]['vn_4_err'][io] for lab in cent_labels])
            axes[0].errorbar(cent_x, vn4, yerr=err4, fmt='^:', ms=5,
                             label=f'v{n}{{4}}')

    axes[0].set_xticks(cent_x)
    axes[0].set_xticklabels(cent_labels, rotation=45, fontsize=8)
    axes[0].set_xlabel('Centrality (%)')
    axes[0].set_ylabel(r'$v_n$')
    axes[0].set_title(r'Integrated $v_n$ vs centrality')
    axes[0].legend(fontsize=7)

    # pT-differential v2{2} for each centrality
    for (lab, res), color in zip(flow.items(), colors):
        pt  = res['pt_cents']
        io  = 0   # n=2
        v2  = res['vn_2_pt'][:, io]
        err = res['vn_2_pt_err'][:, io]
        ok  = np.isfinite(v2)
        if ok.sum() > 0:
            axes[1].errorbar(pt[ok], v2[ok], yerr=err[ok],
                             fmt='o-', ms=3, color=color, label=f'{lab}%')

    axes[1].set_xlabel(r'$p_T$ (GeV)')
    axes[1].set_ylabel(r'$v_2\{2\}(p_T)$')
    axes[1].set_title(r'$v_2\{2\}(p_T)$ by centrality')
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('SanityPlots/flow.pdf')
    print("Saved SanityPlots/flow.pdf")