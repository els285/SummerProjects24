# %%
import uproot
import vector
import numpy as np
import awkward as ak
import matplotlib as mpl
import pandas as pd

# %%
#open root files and make arrays
#edit this to get the right path
path_recon = "/Users/broad/Documents/Documents/university/summer work/neural_network_jet_lep/ttbar_for_DNN_reco.root:Reco"
path_to_save_file = "/Users/broad/Documents/Documents/university/summer work/neural_network_jet_lep/"


tree_recon = uproot.open(path_recon)


# %%
#RECON construction of data leptons
muon_pt = tree_recon["Muon.PT"].array(library='ak')
electron_pt = tree_recon["Electron.PT"].array(library='ak')
lepton_pt = ak.concatenate([muon_pt, electron_pt], axis = 1)



muon_charge = tree_recon["Muon.Charge"].array(library='ak')
electron_charge = tree_recon["Electron.Charge"].array(library='ak')
lepton_charge = ak.concatenate([muon_charge, electron_charge], axis = 1)

muon_Eta = tree_recon["Muon.Eta"].array(library='ak')
electron_Eta = tree_recon["Electron.Eta"].array(library='ak')
lepton_Eta = ak.concatenate([muon_Eta, electron_Eta], axis = 1)

muon_Phi = tree_recon["Muon.Phi"].array(library='ak')
electron_Phi = tree_recon["Electron.Phi"].array(library='ak')
lepton_Phi = ak.concatenate([muon_Phi, electron_Phi], axis = 1)

muon_Phi = tree_recon["Muon.Phi"].array(library='ak')
electron_Phi = tree_recon["Electron.Phi"].array(library='ak')
lepton_Phi = ak.concatenate([muon_Phi, electron_Phi], axis = 1)

muon_m = (tree_recon["Muon.T"].array(library='ak') < 1) * 0.1057
muon_m = ak.values_astype(muon_m, np.float32)
electron_m = (tree_recon["Electron.T"].array(library='ak')<1)*0.511e-3
electron_m = ak.values_astype(electron_m, np.float32)
lepton_m = ak.concatenate([muon_m, electron_m], axis = 1)

#remove events that dont have 2 leptons with opposite charge
recon_lepton_charge_mask = (ak.num(lepton_charge, axis = 1) ==2) & (ak.sum(lepton_charge, axis=1) == 0)
lepton_pt[recon_lepton_charge_mask]

# %%
#RECON construction of data

recon_btagged_jets = tree_recon["Jet.BTag"].array(library='ak')
############this will need to be editted if wanting to include different numbers of b jet ###currently set to only events including 2 bjets
at_least_one_btag_recon_mask = (np.sum(recon_btagged_jets, axis=1) == 2)

#THIS ONE INCLUDES ALL THE CUTS ON THE RECON DATA INCLUDING THE LEPTON CUTS
total_recon_mask = at_least_one_btag_recon_mask & recon_lepton_charge_mask


jet_btagged=recon_btagged_jets[total_recon_mask]

#used to set a parameter array value to 0 if not b-tagged

filtered_jet_pt = tree_recon["Jet.PT"].array(library='ak')[total_recon_mask][jet_btagged==1]
filtered_jet_phi = tree_recon["Jet.Phi"].array(library='ak')[total_recon_mask][jet_btagged==1]
filtered_jet_eta = tree_recon["Jet.Eta"].array(library='ak')[total_recon_mask][jet_btagged==1]
filtered_jet_m = tree_recon["Jet.Mass"].array(library='ak')[total_recon_mask][jet_btagged==1]

recon_jets=vector.zip({'pt': filtered_jet_pt, 'eta': filtered_jet_eta, 'phi': filtered_jet_phi, 'm': filtered_jet_m})
recon_leptons=vector.zip({'pt': lepton_pt, 'eta': lepton_Eta, 'phi': lepton_Phi, 'm': lepton_m})[total_recon_mask]

#aribtarily assign one jet to b and the other jet to bbar this will be sorted by the machine learning
recon_b = recon_jets[:,0]
recon_bbar = recon_jets[:,1]


# %%
#keep track of recon lepton charge
recon_lepton_charge = lepton_charge[total_recon_mask]
recon_lepton_charge_mask_1 = (ak.num(recon_lepton_charge, axis = 1) ==2)
recon_lepton_charge_mask_2 = ~(ak.is_none(recon_lepton_charge))
recon_lepton_charge_mask = recon_lepton_charge_mask_1  & recon_lepton_charge_mask_2
recon_lepton_charge = recon_lepton_charge[recon_lepton_charge_mask]

#defining positive and negative recon leptons
recon_positive_mask = (recon_lepton_charge == 1)
recon_negative_mask = (recon_lepton_charge==-1)
recon_lep_positive = recon_leptons[recon_positive_mask]
recon_lep_negative = recon_leptons[recon_negative_mask]


####FOR SOME REASON NECESSARY FOR FORMATTING################
recon_lep_positive = vector.zip({'pt': [item for sublist in recon_lep_positive.pt for item in sublist], 'eta': [item for sublist in recon_lep_positive.eta for item in sublist], 'phi': [item for sublist in recon_lep_positive.phi for item in sublist], 'e': [item for sublist in recon_lep_positive.e for item in sublist]})
recon_lep_negative = vector.zip({'pt': [item for sublist in recon_lep_negative.pt for item in sublist], 'eta': [item for sublist in recon_lep_negative.eta for item in sublist], 'phi': [item for sublist in recon_lep_negative.phi for item in sublist], 'e': [item for sublist in recon_lep_negative.e for item in sublist]})
############thought these two lines were needed but apparently not now###########
#recon_b = vector.zip({'pt': [item for sublist in recon_b.pt for item in sublist], 'eta': [item for sublist in recon_b.eta for item in sublist], 'phi': [item for sublist in recon_b.phi for item in sublist], 'e': [item for sublist in recon_b.e for item in sublist]})
# recon_bbar = vector.zip({'pt': [item for sublist in recon_bbar.pt for item in sublist], 'eta': [item for sublist in recon_bbar.eta for item in sublist], 'phi': [item for sublist in recon_bbar.phi for item in sublist], 'e': [item for sublist in recon_bbar.e for item in sublist]})
# %%
#we have the signal: lep_positive with b and lep_negative with bbar (since the top goes to b and antitop goes to bbar)
#the backgroung is: lep_positive with bbar and lep_negative with b  (this just wouldnt make sense)
#NOW MAKE OBSERVABLES TO PUT INTO NN

#detla R
signal_delta_r_blep_positive = recon_lep_positive.deltaR(recon_b)
signal_delta_r_bbarlep_negative = recon_lep_negative.deltaR(recon_bbar)
signal_delta_r = np.concatenate((signal_delta_r_bbarlep_negative, signal_delta_r_blep_positive))

#bl invariant mass
signal_invmass_blep_positive = (recon_b+recon_lep_positive).m
signal_invmass_bbarlep_negative = (recon_bbar+recon_lep_negative).m
signal_bl_invmass = np.concatenate((signal_invmass_bbarlep_negative, signal_invmass_blep_positive))

#delta eta 
signal_delta_eta_blep_positive = recon_lep_positive.deltaeta(recon_b)
signal_delta_eta_bbarlep_negative = recon_lep_negative.deltaeta(recon_bbar)
signal_delta_eta = np.concatenate((signal_delta_eta_bbarlep_negative, signal_delta_eta_blep_positive))

#combined pt of bl
signal_combined_pt_blep_positive = (recon_lep_positive+recon_b).pt
signal_combined_pt_bbarlep_negative = (recon_lep_negative+recon_bbar).pt
signal_combined_pt = np.concatenate((signal_combined_pt_bbarlep_negative, signal_combined_pt_blep_positive))
#delta_r between clusters
signal_cluster_delta_r_blep_positive = (recon_lep_positive+recon_b).deltaR(recon_lep_negative+recon_bbar)
signal_cluster_delta_r_bbarlep_negative = (recon_lep_positive+recon_b).deltaR(recon_lep_negative+recon_bbar)
signal_cluster_delta_r = np.concatenate((signal_cluster_delta_r_bbarlep_negative, signal_cluster_delta_r_blep_positive))

#delta_eta between clusters
signal_cluster_delta_eta_blep_positive = (recon_lep_positive+recon_b).deltaeta(recon_lep_negative+recon_bbar)
signal_cluster_delta_eta_bbarlep_negative = (recon_lep_positive+recon_b).deltaeta(recon_lep_negative+recon_bbar)
signal_cluster_delta_eta = np.concatenate((signal_cluster_delta_eta_bbarlep_negative, signal_cluster_delta_eta_blep_positive))


#magnitude of delta_eta + delta_R
signal_mag_de_dr_blep_positive = np.abs(signal_delta_eta_blep_positive + signal_delta_r_blep_positive)
signal_mag_de_dr_bbarlep_negative = np.abs(signal_delta_eta_bbarlep_negative + signal_delta_r_bbarlep_negative)
signal_mag_de_dr = np.concatenate((signal_mag_de_dr_bbarlep_negative, signal_mag_de_dr_blep_positive))
# %%
#convert to h file
import uproot
import h5py
import numpy as np
from tqdm import tqdm

h5_file = h5py.File(path_to_save_file + "data.h5", 'w')
group = h5_file.create_group('Objects')
group.create_dataset("delta_r", data=signal_delta_r)
group.create_dataset("bl_invmass", data=signal_bl_invmass)
group.create_dataset("delta_eta", data=signal_delta_eta )
group.create_dataset("bl_pt", data=signal_combined_pt)
group.create_dataset("cluster_delta_r", data=signal_cluster_delta_r)
group.create_dataset("cluster_delta_eta", data=signal_cluster_delta_eta)
group.create_dataset("mag_de_dr", data=signal_mag_de_dr)
h5_file.close()
print("done")

