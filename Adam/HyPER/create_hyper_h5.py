import uproot 
import numpy as np
from tqdm import tqdm
import h5py
import awkward as ak
import vector

import sys


# path = 'C:/Users/HP/Documents/data/'
filepath = sys.argv[1]
outfile  = sys.argv[2]
pad_to_jet = 20
print('opening root')
root_file   = uproot.open(filepath)['Delphes']


njet = root_file['Jet'].array()        
mask_jet = njet>=6      
njet = njet[mask_jet]         
jet_pt = root_file['Jet/Jet.PT'].array()[mask_jet]
jet_eta = root_file['Jet/Jet.Eta'].array()[mask_jet]
jet_phi = root_file['Jet/Jet.Phi'].array()[mask_jet]     
jet_m = root_file['Jet/Jet.Mass'].array()[mask_jet]
jet_btag = root_file['Jet/Jet.BTag'].array()[mask_jet]

n_particles = root_file['Particle'].array()[mask_jet]
particle_pid = root_file['Particle/Particle.PID'].array()[mask_jet]
particle_status = root_file['Particle/Particle.Status'].array()[mask_jet]
particle_m = root_file['Particle/Particle.Mass'].array()[mask_jet] 
particle_pt = root_file['Particle/Particle.PT'].array()[mask_jet] 
particle_eta = root_file['Particle/Particle.Eta'].array()[mask_jet]  
particle_phi = root_file['Particle/Particle.Phi'].array()[mask_jet] 

length = len(n_particles)
new_values = np.array([1,4,2,3,5,6])

def pad_variable(variable, max_len, pad_to = 0):
    padded_variable = ak.pad_none(variable, max_len, axis=1)
    return ak.fill_none(padded_variable, pad_to)

def create_padded_vector(pt, eta, phi, m, max_len, pad_eta=0):
    padded_pt = pad_variable(pt, max_len)
    padded_eta = pad_variable(eta, max_len, pad_to = pad_eta)
    padded_phi = pad_variable(phi, max_len)
    padded_m = pad_variable(m, max_len)
    return vector.zip({'pt':padded_pt, 'eta':padded_eta, 'phi':padded_phi, 'm' :padded_m})


def quick_method(jets, particles, max_jet, limit = length):
    # requires padded jet array
    matched = np.zeros((limit,max_jet))
    mask = np.zeros((limit, max_jet), dtype=bool)
    for i in range(max_jet):
        delta_Rs = jets[:,i][:limit].deltaR(particles[:limit])  
        min_R = np.min(delta_Rs, axis=1)
        index = new_values[np.argmin(delta_Rs, axis=1)]
        matched[:,i] = np.where(min_R<0.4, index,0)
    return matched

def expensive_method(jets, particles, max_jet ,limit=length):
    # requires non padded jet array
    limit = min(len(jets), limit)
    print(max_jet)
    matched = np.zeros((limit,max_jet))
    for event, (event_jets, event_particles) in tqdm(enumerate(zip(jets[:limit], particles[:limit]))):
        all_Rs = np.array([jet.deltaR(event_particles) for jet in event_jets])
        mins = np.zeros((max_jet))
        for i in range(6):
            arg = np.argmin(all_Rs)
            mins[arg//6] = np.where(all_Rs[arg//6, arg%6]<0.4, new_values[arg%6], 0)
            all_Rs[:,arg%6] = 99
            all_Rs[arg//6,:] = 99
        matched[event] = mins
    return matched


def find_repeats(data):
    mask = np.full(len(data), False)
    for i in range(1,7):
        value = np.sum(data == i, axis=1)
        mask = mask | (value>1)
    #jet_truth_matched[mask]
    return np.arange(len(data))[mask]

def crop(array, n_jet, max_jet):
    mask = np.zeros(np.shape(array), dtype=bool)
    for i in range(max_jet):
        mask[:,i]= n_jet>i
    return ak.drop_none(ak.mask(array,mask))



def full_method(jet_vectors, jet_vectors_padded, particle_vectors, max_jet, limit = length, crop=False):
    jet_truth_matched = quick_method(jet_vectors_padded, particle_vectors, max_jet, limit)
    bad_indicies = find_repeats(jet_truth_matched)
    print(f'{len(bad_indicies)} events to recalculate')
    jet_truth_matched[bad_indicies] = expensive_method(jet_vectors[bad_indicies], particle_vectors[bad_indicies], max_jet)
    if crop:
        return crop(jet_truth_matched, njet[:limit], max_jet)
    return jet_truth_matched

print('creating vectors')   
max_jet = np.max(njet)
particle_vectors = vector.zip({'pt':particle_pt, 'eta':particle_eta, 'phi':particle_phi, 'm' :particle_m})[particle_status == 23][:,-6:]
jet_vectors_padded = vector.zip({'pt':pad_variable(jet_pt, max_jet), 'eta':pad_variable(jet_eta, max_jet, pad_to = 99), 'phi':pad_variable(jet_phi, max_jet), 'm' :pad_variable(jet_m, max_jet)})
jet_vectors  = vector.zip({'pt':jet_pt, 'eta':jet_eta, 'phi':jet_phi, 'm' :jet_m})

print('truth matching')
jet_truthmatch = full_method(jet_vectors, jet_vectors_padded, particle_vectors, max_jet)

print('forming data')
data = {}
data['njet'] = njet
data['nbtagged'] = np.sum(jet_btag, axis=1)
data['allMatchedEvent'] = np.sum(jet_truthmatch !=0, axis=1) == njet
data['leastOneMatchedW'] = np.any(np.isin(jet_truthmatch, [2,3,5,6]), axis=1)
data['jet_pt'] = pad_variable(jet_pt, max_jet)
data['jet_eta'] = pad_variable(jet_eta, max_jet, pad_to = 0)
data['jet_phi'] = pad_variable(jet_phi, max_jet)
data['jet_e'] = pad_variable(vector.zip({'pt':jet_pt, 'eta':jet_eta, 'phi':jet_phi, 'm' :jet_m}).e, max_jet)
data['jet_m'] = pad_variable(jet_m, max_jet)
data['jet_btag'] = pad_variable(jet_btag, max_jet)
data['jet_truthmatch'] = jet_truthmatch
data['particle_pid'] = particle_pid[particle_status == 23][:,-6:]
data['particle_pt'] = particle_pt[particle_status == 23][:,-6:]
data['particle_eta'] = particle_eta[particle_status == 23][:,-6:]
data['particle_phi'] = particle_phi[particle_status == 23][:,-6:]
data['particle_m'] = particle_m[particle_status == 23][:,-6:]

print('saving data')
h5_file = h5py.File(outfile, 'w')
group = h5_file.create_group('Delphes')
for key, value in data.items():
    group.create_dataset(key, data=value)

h5_file.close()
print('program finished')