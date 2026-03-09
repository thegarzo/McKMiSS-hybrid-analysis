import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

import glob
import parser as ps
from parameters import *
import processer as pr
import sanityplots as sanity

import matplotlib.cm as cm
import pickle


def main():

    if len(sys.argv) < 3:
        print("Usage: python per_node_analysis.py "
              "<base_path> <output_dir>")
        sys.exit(1)

    base_path           = sys.argv[1]
    output_dir          = sys.argv[2]
    EventPaths = ps.Parser(base_path)
    files   = EventPaths.get_all_h5_paths()
    # print(files)

    sanity_dir= output_dir+"/SanityPlots"
    data_dir= output_dir+"/Data"
    os.makedirs(sanity_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    #assign selection method
    irap= RAP_CUTS_ASSIGMENTS["ALICE midrapidity"]
    print("irap=",irap)
    records = pr.first_pass_centrality(files, irap_cent=irap)


    print(records['dNch_deta'])
    # quick look at the distribution

    # make centrality masks and print info about the bins
    bin_edges=[0,2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    mask, infos = pr.make_centrality_masks(records, bins_percentile=bin_edges)

    ####  PLOT PLOT PLOT PLOT  ####
    sanity.plot_events_selected(infos, sanity_dir,records)
    ####  PLOT PLOT PLOT PLOT  ####


    ## Let's compute spectra now, where we will do it for mid-rapidity in ALICE
    irap= RAP_CUTS_ASSIGMENTS["ALICE midrapidity"]
    spectra ={}
    CHADS="charged_hadrons"
    print(f"Processing pt-spectra for {CHADS} ...")
    spectra[CHADS]=pr.compute_spectra(records, mask, irap, CHADS)
    
    ####  SAVE DICTIONARY  ####
    with open(data_dir+'/spectra_CHADS.pkl', 'wb') as f:
        pickle.dump(spectra, f)

    ####  PLOT PLOT PLOT PLOT  ####
    sanity.plot_dNch_eta_cent(spectra,sanity_dir)
    sanity.plot_avg_pt_cent_CHADs(spectra,sanity_dir)
    sanity.plot_pt_spectra(spectra,sanity_dir,"charged_hadrons")
    # sanity.plot_pt_spectra(spectra,sanity_dir,"pi_plus")
    # sanity.plot_pt_spectra(spectra,sanity_dir,"kaon_plus")
    # sanity.plot_pt_spectra(spectra,sanity_dir,"proton")

if __name__ == "__main__":
    main()






