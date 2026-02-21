import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os

import glob
from parameters import *
import processer as pr
import sanityplots as sanity

import matplotlib.cm as cm

files   = ['/Users/garcia/Documents/McDIPPER/Tests/FullTest/temp_fldr_202512_4_146427/results/hydro_event_0042.h5']

# irap_cent=2 might be your forward window [2.0, 5.0] for example —
# check your RAP_CUTS definition to confirm which index is which

# print(files)
# pr.inspect_hdf5(files[0])
irap= RAP_CUTS_ASSIGMENTS["ALICE midrapidity"]
print("irap=",irap)
records = pr.first_pass_centrality(files, irap_cent=irap)

print(records['dNch_deta'])
# quick look at the distribution
os.makedirs("SanityPlots", exist_ok=True)

# make centrality masks and print info about the bins
bin_edges=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
mask, infos = pr.make_centrality_masks(records, bins_percentile=bin_edges)

####  PLOT PLOT PLOT PLOT  ####
sanity.plot_events_selected(infos, records)
####  PLOT PLOT PLOT PLOT  ####

dnch=pr.compute_dNch_deta(mask, records, species='charged_hadrons')
####  PLOT PLOT PLOT PLOT  ####
sanity.plot_dNch_deta(dnch)
####  PLOT PLOT PLOT PLOT  ####


## Let's compute spectra now, where we will do it for mid-rapidity in ALICE
irap= RAP_CUTS_ASSIGMENTS["ALICE midrapidity"]
spectra ={}
for species in SPECIES.keys():
    print(f"Processing pt-spectra for {species} ...")
    spectra[species]=pr.compute_spectra(records, mask, irap, species)

####  PLOT PLOT PLOT PLOT  ####
sanity.plot_pt_spectra(spectra,"charged_hadrons")
sanity.plot_pt_spectra(spectra,"pi_plus")
sanity.plot_pt_spectra(spectra,"proton")
####  PLOT PLOT PLOT PLOT  ####



## Flow 
# signal: pi_plus at midrapidity
# reference: charged_hadrons at forward rapidity (different window -> no autocorrelations)
irap_sig = RAP_CUTS_ASSIGMENTS['ALICE midrapidity']   # 0: [-0.8, 0.8]
irap_ref = RAP_CUTS_ASSIGMENTS['VZERO-A']             # 2: [2.8, 5.1]

# flow = pr.compute_flow(mask, records,irap_sig,irap_ref,'pi_plus','charged_hadrons')

flow = pr.compute_flow_cumulants(
    masks       = mask,
    records     = records,
    irap_mid    = RAP_CUTS_ASSIGMENTS['ALICE midrapidity'],  # 0
    irap_subA   = RAP_CUTS_ASSIGMENTS['ALICE Flow Sub-event A'],  # 4
    irap_subB   = RAP_CUTS_ASSIGMENTS['ALICE Flow Sub-event B'],  # 5
    species     = 'charged_hadrons',
    ref_species = 'charged_hadrons',
)

####  PLOT PLOT PLOT PLOT  ####
sanity.plot_flows(flow)
####  PLOT PLOT PLOT PLOT  ####