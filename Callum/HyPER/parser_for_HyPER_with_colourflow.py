# %%
import uproot 
import numpy as np
from tqdm import tqdm
import h5py
import awkward as ak
import vector
import sys
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)


# path = '/mnt/c/Users/HP/Documents/data/HyPER'
# os.chdir(path)
filepath = "C:/Users/broad/Downloads/skim_3107_CF_3.root"
outfile  = "output.h5"
pad_to_jet = 20
new_values = np.array([1,4,2,3,5,6])
#new_values = np.array([1,7,4,10,2,3,8,9,5,6,11,12])
p_status_len = len(new_values)
print('opening root')
root_file   = uproot.open(filepath)['Delphes']

jet_pt   = root_file['Jet/Jet.PT'].array()
jet_eta  = root_file['Jet/Jet.Eta'].array()
cuts = np.logical_and(jet_pt > 25 , abs(jet_eta) < 2.5)
#Apply cuts of jet_pt > 25GeV and absolute eta < 2.5


# %%

jet_pt = jet_pt[cuts]
jet_eta = jet_eta[cuts]
jet_phi  = root_file['Jet/Jet.Phi'].array()[cuts]     
jet_m    = root_file['Jet/Jet.Mass'].array()[cuts]
jet_btag = root_file['Jet/Jet.BTag'].array()[cuts]

#ensure that there are at least 6 jets in each event
njet = ak.num(jet_btag, axis=1)
particle_status = root_file['Particle/Particle.Status'].array()
jet_cut = (njet >=6) & (np.sum(particle_status == 23, axis=1)>=p_status_len)
particle_status = particle_status[jet_cut]
njet = njet[jet_cut]
jet_pt = jet_pt[jet_cut]
jet_eta = jet_eta[jet_cut]
jet_phi = jet_phi[jet_cut]
jet_m = jet_m[jet_cut]
jet_btag = jet_btag[jet_cut]

n_particles     = root_file['Particle'].array()[jet_cut]
particle_pid    = root_file['Particle/Particle.PID'].array()[jet_cut]
particle_m      = root_file['Particle/Particle.Mass'].array()[jet_cut] 
particle_pt     = root_file['Particle/Particle.PT'].array()[jet_cut] 
particle_eta    = root_file['Particle/Particle.Eta'].array()[jet_cut]  
particle_phi    = root_file['Particle/Particle.Phi'].array()[jet_cut] 

length = len(n_particles)

#get track data
track_pt  = root_file['Track/Track.PT'].array()[jet_cut]     
track_m    = root_file['Track/Track.Mass'].array()[jet_cut]
track_eta  = root_file['Track/Track.Eta'].array()[jet_cut]  
track_phi    = root_file['Track/Track.Phi'].array()[jet_cut]

# %%




def pad_variable(variable, max_len, pad_to = 0):
    padded_variable = ak.pad_none(variable, max_len, axis=1, clip=True)
    return ak.fill_none(padded_variable, pad_to)

def create_padded_vector(pt, eta, phi, m, max_len, pad_eta=0):
    padded_pt = pad_variable(pt, max_len)
    padded_eta = pad_variable(eta, max_len, pad_to = pad_eta)
    padded_phi = pad_variable(phi, max_len)
    padded_m = pad_variable(m, max_len)
    return vector.zip({'pt':padded_pt, 'eta':padded_eta, 'phi':padded_phi, 'm' :padded_m})


def quick_method(jets, particles, max_jet):
    # requires padded jet array
    matched = np.zeros((length,max_jet))
    for i in range(max_jet):
        delta_Rs = jets[:,i].deltaR2(particles)  
        min_R = np.min(delta_Rs, axis=1)
        index = new_values[np.argmin(delta_Rs, axis=1)]
        matched[:,i] = np.where(min_R<0.16, index,0)
    return matched

def expensive_method(jets, particles, max_jet):
    # requires non padded jet array
    matched = np.zeros((len(jets),max_jet))
    for event, (event_jets, event_particles) in tqdm(enumerate(zip(jets, particles))):
        all_Rs = np.array([jet.deltaR2(event_particles) for jet in event_jets])
        mins = np.zeros((max_jet))
        for i in range(p_status_len):
            arg = np.argmin(all_Rs)
            mins[arg//p_status_len] = np.where(all_Rs[arg//p_status_len, arg%p_status_len]<0.16, new_values[arg%p_status_len], 0)
            all_Rs[:,arg%p_status_len] = 99
            all_Rs[arg//p_status_len,:] = 99
        matched[event] = mins
    return matched


def find_repeats(data):
    mask = np.full(len(data), False)
    for i in range(1,p_status_len+1):
        value = np.sum(data == i, axis=1)
        mask = mask | (value>1)
    #jet_truth_matched[mask]
    return np.arange(len(data))[mask]



def full_method(jet_vectors, jet_vectors_padded, particle_vectors, max_jet):
    jet_truth_matched = quick_method(jet_vectors_padded, particle_vectors, max_jet)
    bad_indicies = find_repeats(jet_truth_matched)
    print(f'{len(bad_indicies)} events to recalculate')
    jet_truth_matched[bad_indicies] = expensive_method(jet_vectors[bad_indicies], particle_vectors[bad_indicies], max_jet)
    return jet_truth_matched

def pull1(jet, tracks):
    pt_track = tracks.pt
    pt_jet = jet.pt
    delta_y = tracks.rapidity - jet.rapidity
    delta_phi =  tracks.deltaphi(jet)
    r = np.sqrt(delta_y**2+delta_phi**2)
    pull_y = ((pt_track * r) / pt_jet) * delta_y
    pull_phi = ((pt_track * r) / pt_jet) * delta_phi
    x= vector.zip({'x': pull_y, 'y': pull_phi})
    return np.sum(x, axis=1)

def theta_p(jet1,jet2, pulls):
    jcv12 = vector.zip({'x': jet2.rapidity - jet1.rapidity,
                        'y': jet2.deltaphi(jet1)})

    return np.arctan2( jcv12.x*pulls.y - jcv12.y*pulls.x, jcv12.x*pulls.x + jcv12.y*pulls.y)


# %%

# Building vectors
print('creating vectors')   
particle_vectors   = vector.zip({'pt':particle_pt, 'eta':particle_eta, 'phi':particle_phi, 'm' :particle_m})[particle_status == 23][:,-p_status_len:]
jet_vectors_padded = vector.zip({'pt':pad_variable(jet_pt, pad_to_jet), 'eta':pad_variable(jet_eta, pad_to_jet, pad_to = 99), 'phi':pad_variable(jet_phi, pad_to_jet), 'm' :pad_variable(jet_m, pad_to_jet)})
track_vectors = vector.zip({'pt':track_pt, 'eta':track_eta, 'phi':track_phi, 'm' :track_m})
jet_vectors        = vector.zip({'pt':jet_pt, 'eta':jet_eta, 'phi':jet_phi, 'm' :jet_m})[:,:pad_to_jet]

# Performing truth-matching
print('truth matching')
jet_truthmatch = full_method(jet_vectors, jet_vectors_padded, particle_vectors, pad_to_jet)
jet_id         = 1*(ak.local_index(jet_truthmatch) < njet[:,None]) + 0*(ak.local_index(jet_truthmatch) > njet[:,None])
jet_truthmatch = jet_truthmatch + -9*(ak.local_index(jet_truthmatch) > njet[:,None])


# %%
print("matching tracks to correct jets")
tracks_in_jet={}
for i in tqdm(range (20)):
    tracks_in_jet[i] = track_vectors[(jet_vectors_padded[:,i].deltaR(track_vectors) <= 0.4)&(jet_vectors_padded[:,i].deltaR(track_vectors)<=jet_vectors_padded[:,0].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,1].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,2].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,3].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,4].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,5].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,6].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,7].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,8].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,9].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,10].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,11].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,12].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,13].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,14].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,15].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,16].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,17].deltaR(track_vectors))&(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,18].deltaR(track_vectors)) &(jet_vectors_padded[:,i].deltaR(track_vectors) <=jet_vectors_padded[:,19].deltaR(track_vectors))]
    

# %%
print("calculating pull of each jet")
jet_pull={}
for i in tqdm(range (20)):
    jet_pull[i]=pull1(jet_vectors_padded[:,i],tracks_in_jet[i])
jet_pull_rapidity = np.column_stack([jet_pull[i].x for i in range(0, 20)])
jet_pull_phi= np.column_stack([jet_pull[i].y for i in range(0, 20)])


# %%

print('forming data')

# Fill global data
global_dt   = np.dtype([('njet', np.float32), ('nbTagged', np.float32)])
global_data = np.zeros((len(njet), 1), dtype=global_dt)
njet     = njet.to_numpy().reshape(-1,1)
nbTagged = np.sum(jet_btag, axis=1).to_numpy().reshape(-1,1)

global_data['njet']     = njet
global_data['nbTagged'] = nbTagged

# Fill jet data
node_dt  = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('pull_rapidity', np.float32), ('pull_phi', np.float32),('rapidity', np.float32), ('btag', np.float32), ('charge', np.float32), ('id', np.float32)])
jet_data = np.zeros((len(njet), pad_to_jet), dtype=node_dt)

jet_data['pt']   = jet_vectors_padded.pt
jet_data['eta']  = pad_variable(jet_eta, pad_to_jet, pad_to = 0)
jet_data['phi']  = jet_vectors_padded.phi
jet_data['e']    = jet_vectors_padded.e
jet_data['btag'] = pad_variable(jet_btag, pad_to_jet)
jet_data['charge'] = np.zeros(len(njet)).reshape(-1,1)
jet_data['id']     = jet_id
jet_data["pull_rapidity"] = jet_pull_rapidity
jet_data["pull_phi"] = jet_pull_phi
jet_data["rapidity"] = jet_vectors_padded.rapidity

VertexID_data = jet_truthmatch
IndexSelect = 1*(np.sum(np.isin(VertexID_data, [1,2,3,4,5,6]), axis=1) == 6)



# %%
print('saving data')
h5_file = h5py.File(outfile, 'w')
inputs_group = h5_file.create_group('INPUTS')
labels_group = h5_file.create_group('LABELS')

inputs_group.create_dataset("jet", data=jet_data)
inputs_group.create_dataset("global", data=global_data)
labels_group.create_dataset("VertexID", data=VertexID_data.to_numpy().astype(np.int64))
labels_group.create_dataset("IndexSelect", data = IndexSelect.to_numpy().astype(np.int32))

h5_file.close()
print('program finished')


# %%
h5_file = h5py.File(outfile, 'r')
inputs = h5_file["INPUTS"]
labels = h5_file["LABELS"]   
labels.keys()
h5_file.close()



