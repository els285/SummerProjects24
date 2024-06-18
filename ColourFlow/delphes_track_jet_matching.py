from tqdm import tqdm
import uproot
import numpy as np


# Useful functions
nested_hstack = lambda L:  [q for p in L for q in p]
DeltaR        = lambda X,Y: (X**2 + Y**2)**(0.5)

def matching(list_of_track_eta,list_of_track_phi,jet_eta,jet_phi):

    r"""Returns the matched indices of the tracks corresponding to the jet in question
    Args:
        list_of_track_eta: array of all track eta in event
        list_of_track_phi: array of all track phi in event
        jet_eta: eta of jet in question
        jet_phi: phi of jet in question
    """
    
    eta_diff = list_of_track_eta - jet_eta*np.ones(len(list_of_track_eta))
    phi_diff = list_of_track_phi - jet_phi*np.ones(len(list_of_track_phi))
    DR_array = DeltaR(eta_diff,phi_diff)
    return np.argwhere(DR_array<0.4)[:,0]


def process(jet_tree,track_tree):
    
    r""" Applies the matching  by looping over the events in the trees
    
    Args:
        jet_tree:   awkward/numpy array of jet branches
        track_tree: awkward/numpy array of track branches
    """

    output_dic = {}
    Nevents = len(jet_tree)
    for eN in tqdm(range(Nevents)):
        inner_dic = {}
        Njets = len(jet_tree[eN]["Jet.Eta"])
        track_Eta_arr = track_tree["Track.Eta"][eN]
        track_Phi_arr = track_tree["Track.Phi"][eN]
        for jet_index in range(Njets):
            jet_eta = jet_tree[eN]["Jet.Eta"][jet_index]
            jet_phi = jet_tree[eN]["Jet.Phi"][jet_index]
            matched_indices = matching(list_of_track_eta=track_Eta_arr,
                    list_of_track_phi=track_Phi_arr,
                    jet_eta=jet_eta,jet_phi=jet_phi)
            inner_dic[jet_index] = {"jet_pt"    :   jet_tree[eN]["Jet.PT"][jet_index],
                                    "jet_eta"   :   jet_tree[eN]["Jet.Eta"][jet_index],
                                    "jet_phi"   :   jet_tree[eN]["Jet.Phi"][jet_index],
                                    "jet_mass"  :   jet_tree[eN]["Jet.Mass"][jet_index],
                                    "track_pt"  :   track_tree[eN]["Track.PT"][matched_indices],
                                    "track_eta" :   track_tree[eN]["Track.Eta"][matched_indices],
                                    "track_phi" :   track_tree[eN]["Track.Phi"][matched_indices],
                                    "track_mass" :   track_tree[eN]["Track.Mass"][matched_indices]
                                    }
        output_dic[eN] = inner_dic   
    return output_dic

def restructure_data(output_dic):
    
    r"""This just rearranges the output of process"""

    rearranged = {  "Njets"     : [],
                    "jet_pt"    : [],
                    "jet_eta"   : [],
                    "jet_phi"   : [],
                    "jet_mass"  : [],
                    "track_pt"  : [],
                    "track_eta" : [],
                    "track_phi" : [],
                    "track_mass": []}
    for event in output_dic.values():
        jet_pt      = []
        jet_eta     = []
        jet_phi     = []
        jet_mass    = []
        track_pt    = []
        track_eta   = []
        track_phi   = []
        track_mass  = []
        rearranged["Njets"].append(len(event))
        for jet in event.values():
            jet_pt.append(jet["jet_pt"])
            jet_eta.append(jet["jet_eta"])
            jet_phi.append(jet["jet_phi"])
            jet_mass.append(jet["jet_mass"])
            track_pt.append(jet["track_pt"])
            track_eta.append(jet["track_eta"])
            track_phi.append(jet["track_phi"])
            track_mass.append(jet["track_mass"])
        rearranged["jet_pt"].append(jet_pt)
        rearranged["jet_eta"].append(jet_eta)
        rearranged["jet_phi"].append(jet_phi)
        rearranged["jet_mass"].append(jet_mass)
        rearranged["track_pt"].append(track_pt)
        rearranged["track_eta"].append(track_eta)
        rearranged["track_phi"].append(track_phi)
        rearranged["track_mass"].append(track_mass)
        
        return rearranged
    
    
def do_matching(jet_tree,track_tree) -> dict:
    
    r"""Run the process for matching tracks to jets using a Delta(R) of 0.4 and
     restructures the data
    
    Args:
        jet_tree:   awkward/numpy array of jet branches
        track_tree: awkward/numpy array of track branches
    """
    
    if len(jet_tree)!=len(track_tree):
        raise ValueError("The loaded trees are of different length")
    matched_data_dic = process(jet_tree,track_tree)
    return restructure_data(matched_data_dic)