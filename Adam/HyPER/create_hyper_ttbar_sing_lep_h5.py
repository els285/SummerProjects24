import uproot 
import numpy as np
from tqdm import tqdm
import h5py
import awkward as ak
import vector
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# path = '/mnt/c/Users/HP/Documents/data/HyPER/'
# os.chdir(path)
filepath = sys.argv[1]
outfile  = sys.argv[2]

pad_to_jet = 17
pad_lepton = 3
new_values = np.array([1,4,2,3,5,6])
p_status_len = 4

print('opening root')
root_file   = uproot.open(filepath)['Delphes']

print('finding jet data')
jet_pt   = root_file['Jet/Jet.PT'].array()
jet_eta  = root_file['Jet/Jet.Eta'].array()
jet_m    = root_file['Jet/Jet.Mass'].array()
# cuts = np.logical_and(jet_pt > 25 , abs(jet_eta) < 2.5, jet_m>0)
cuts = (jet_pt > 25) & (abs(jet_eta) < 2.5) & (jet_m>0)
#Apply cuts of jet_pt > 25GeV and absolute eta < 2.5

jet_pt = jet_pt[cuts]
jet_eta = jet_eta[cuts]
jet_m = jet_m[cuts]
jet_phi  = root_file['Jet/Jet.Phi'].array()[cuts]     
jet_btag = root_file['Jet/Jet.BTag'].array()[cuts]

#ensure that there are at least 6 jets in each event
njet = ak.num(jet_btag, axis=1)
jet_cut = (njet >=4)
njet = njet[jet_cut]
jet_pt = jet_pt[jet_cut]
jet_eta = jet_eta[jet_cut]
jet_phi = jet_phi[jet_cut]
jet_m = jet_m[jet_cut]
jet_btag = jet_btag[jet_cut]

print('finding particle data')
particle_pid = root_file['Particle/Particle.PID'].array()[jet_cut]
n_particles     = ak.num(particle_pid, axis=1)
particle_status = root_file['Particle/Particle.Status'].array()[jet_cut]
particle_m      = root_file['Particle/Particle.Mass'].array()[jet_cut]
particle_pt     = root_file['Particle/Particle.PT'].array()[jet_cut]
particle_eta    = root_file['Particle/Particle.Eta'].array()[jet_cut]
particle_phi    = root_file['Particle/Particle.Phi'].array()[jet_cut]

print('finding lepton data')
n_el    = root_file['Electron'].array()[jet_cut] 
el_pt    = root_file['Electron/Electron.PT'].array()[jet_cut]
el_eta    = root_file['Electron/Electron.Eta'].array()[jet_cut]
el_phi = ak.full_like(el_pt, 0)
el_m = ak.full_like(el_pt, 0.000511)

n_mu   = root_file['Electron'].array()[jet_cut] 
mu_pt    = root_file['Muon/Muon.PT'].array()[jet_cut]
mu_eta    = root_file['Muon/Muon.Eta'].array()[jet_cut] 
mu_phi    = root_file['Muon/Muon.Phi'].array()[jet_cut]
mu_m = ak.full_like(mu_pt, 0.10566)

print('finding met data')
n_met = root_file['MissingET'].array()[jet_cut]
met_met = root_file['MissingET/MissingET.MET'].array()[jet_cut]
met_eta = root_file['MissingET/MissingET.Eta'].array()[jet_cut]
met_phi = root_file['MissingET/MissingET.Phi'].array()[jet_cut]

length = len(n_particles)
particle_cut = (np.abs(particle_pid) == 11) | (np.abs(particle_pid) == 13)

def pad_variable(variable, max_len, pad_to = 0):
    padded_variable = ak.pad_none(variable, max_len, axis=1, clip=True)
    return ak.fill_none(padded_variable, pad_to)

def quick_method(jets, particles, max_jet, w_minus):
    matched = np.zeros((length,max_jet))
    for i in range(max_jet):
        delta_Rs = jets[:,i].deltaR2(particles)  
        min_R = np.min(delta_Rs, axis=1)
        arg = np.argmin(delta_Rs, axis=1)
        index = new_values[arg]
        matched[:,i] = np.where(min_R<0.16, index,0)
    return matched + 3 * (np.isin(matched, [2,3])& w_minus[:,None])

def expensive_method(jets, particles, max_jet, w_minus):
    # requires non padded jet array
    matched = np.zeros((len(jets),max_jet))
    for event, (event_jets, event_particles) in tqdm(enumerate(zip(jets, particles))):
        all_Rs = np.array([jet.deltaR2(event_particles) for jet in event_jets])
        mins = np.zeros((pad_to_jet))
        for i in range(p_status_len):
            arg = np.argmin(all_Rs)
            mins[arg//p_status_len] = np.where(all_Rs[arg//p_status_len, arg%p_status_len]<0.16, new_values[arg%p_status_len], 0)
            all_Rs[:,arg%p_status_len] = 99
            all_Rs[arg//p_status_len,:] = 99
        matched[event] = mins
    return matched + 3 * (np.isin(matched, [2,3])& w_minus[:,None])


def find_repeats(data):
    mask = np.full(len(data), False)
    for i in range(1,p_status_len+1):
        value = np.sum(data == i, axis=1)
        mask = mask | (value>1)
    #jet_truth_matched[mask]
    return np.arange(len(data))[mask]


def full_method(jet_vectors, jet_vectors_padded, particle_vectors, max_jet, w_minus):
    jet_truth_matched = quick_method(jet_vectors_padded, particle_vectors, max_jet, w_minus)
    bad_indicies = find_repeats(jet_truth_matched)
    print(f'{len(bad_indicies)} events to recalculate')
    jet_truth_matched[bad_indicies] = expensive_method(jet_vectors[bad_indicies], particle_vectors[bad_indicies], max_jet, w_minus[bad_indicies])
    return jet_truth_matched

def find_lep_charge(lepton_vector_padded, lepton_truth_vector, lepton_pid_truth):
    matched = np.zeros((length, pad_lepton))
    for i in range(pad_lepton):
        deltaRs = lepton_vector_padded[:,i].deltaR2(lepton_truth_vector)
        min_R = np.min(deltaRs, axis=1)
        index = np.argmin(deltaRs, axis=1)
        pid = lepton_pid_truth[ak.local_index(lepton_pid_truth) == index][:,0]
        matched[:,i] = np.where(min_R<0.16, np.sign(-1*pid), 0)
    return matched

print('matching jets')
particle_vectors   = vector.zip({'pt':particle_pt, 'eta':particle_eta, 'phi':particle_phi, 'm' :particle_m})[particle_status == 23][:,-p_status_len:]
jet_vectors_padded = vector.zip({'pt':pad_variable(jet_pt, pad_to_jet), 'eta':pad_variable(jet_eta, pad_to_jet, pad_to = 99), 'phi':pad_variable(jet_phi, pad_to_jet), 'm' :pad_variable(jet_m, pad_to_jet)})
jet_vectors        = vector.zip({'pt':jet_pt, 'eta':jet_eta, 'phi':jet_phi, 'm' :jet_m})[:,:pad_to_jet]
w_minus = np.array(np.isin(particle_pid[particle_status == 23][:,-2], [1,3]))
jet_labels = full_method(jet_vectors, jet_vectors_padded, particle_vectors, pad_to_jet, w_minus)

print('matching leptons')
lepton_pt = np.concatenate((el_pt, mu_eta), axis=1)
lepton_eta = np.concatenate((el_eta, mu_eta), axis=1)
lepton_phi = np.concatenate((el_phi, mu_phi), axis=1)
lepton_m = np.concatenate((el_m, mu_m), axis=1)
lepton_vector_padded = vector.zip({'pt':pad_variable(lepton_pt, pad_lepton), 'eta':pad_variable(lepton_eta, pad_lepton, 99), 'phi':pad_variable(lepton_phi, pad_lepton), 'm' :pad_variable(lepton_m, pad_lepton)})

lepton_pt_truth = particle_pt[particle_cut]
lepton_eta_truth = particle_eta[particle_cut]
lepton_phi_truth = particle_phi[particle_cut]
lepton_m_truth = particle_m[particle_cut]
lepton_pid_truth = particle_pid[particle_cut]
lepton_truth_vector = vector.zip({'pt':lepton_pt_truth, 'eta':lepton_eta_truth, 'phi':lepton_phi_truth, 'm' :lepton_m_truth})

lepton_charge = find_lep_charge(lepton_vector_padded, lepton_truth_vector, lepton_pid_truth)
positive = lepton_charge == 1
negative = lepton_charge == -1
positive[np.cumsum(positive, axis=1)>1] = False
negative[np.cumsum(negative, axis=1)>1] = False

lepton_labels = 2*(positive & w_minus[:,None]) + 5*(positive & ~w_minus[:,None])
neutrino_labels = 3*w_minus[:,None] + 6*(~w_minus[:,None])
VertexID_data = np.concatenate((jet_labels, lepton_labels, neutrino_labels), axis=1)

neutrino = vector.zip({'pt':pad_variable(met_met, 1), 'eta':pad_variable(met_eta, 1), 'phi':pad_variable(met_phi, 1), 'm' :np.zeros((length, 1))})
print('creating data structures')

global_dt   = np.dtype([('njet', np.float32), ('nbTagged', np.float32)])
global_data = np.zeros((len(njet), 1), dtype=global_dt)
global_data['njet']     = njet[:,None]
global_data['nbTagged'] = np.sum(jet_btag, axis=1)[:,None]

node_dt  = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.float32), ('charge', np.float32), ('id', np.float32)])
jet_data = np.zeros((len(njet), pad_to_jet + pad_lepton+1), dtype=node_dt)

jet_data['pt']   = np.concatenate((jet_vectors_padded.pt, lepton_vector_padded.pt, neutrino.pt), axis=1)
jet_data['eta']  = np.concatenate((pad_variable(jet_eta, pad_to_jet), pad_variable(lepton_eta, pad_lepton), neutrino.eta), axis=1)
jet_data['phi']  = np.concatenate((jet_vectors_padded.phi, lepton_vector_padded.phi, neutrino.phi), axis=1)
jet_data['e']    = np.concatenate((jet_vectors_padded.e, lepton_vector_padded.e, neutrino.e), axis=1)
jet_data['btag'] = np.concatenate((pad_variable(jet_btag, pad_to_jet), np.zeros((length, pad_lepton+1))), axis=1)
jet_data['charge'] = np.concatenate((np.zeros((length, pad_to_jet)), lepton_charge, np.zeros((length,1))), axis=1)
lepton_id = pad_variable(np.concatenate((ak.full_like(el_pt, -1), ak.full_like(mu_pt, -2)),axis=1), pad_lepton)
jet_id = pad_variable(ak.full_like(jet_pt, 1), pad_to_jet)
jet_data['id']     = np.concatenate((jet_id, lepton_id, np.zeros((length, 1))), axis=1)

print('saving data')
h5_file = h5py.File(outfile, 'w')
inputs_group = h5_file.create_group('INPUTS')
labels_group = h5_file.create_group('LABELS')

inputs_group.create_dataset("jet", data=jet_data)
inputs_group.create_dataset("global", data=global_data)
labels_group.create_dataset("VertexID", data=VertexID_data)

h5_file.close()
print('program complete')
