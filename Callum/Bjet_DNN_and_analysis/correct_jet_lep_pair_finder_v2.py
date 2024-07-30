# %%
#takes h5 file of input data and other variables for neural network then:
#   1) selects ONLY event with 2 btagged jets and 2 oppositely charged leptons
#   2) finds the correct lepton/bjet pair (so correctly assigns which jet is b or bbar)

#current accuracy is 82%


#import modules
import h5py
import numpy as np
import vector
import awkward as ak
import pandas as pd  # to store data as dataframe
import matplotlib.pyplot as plt  # for plotting
import torch  # import PyTorch
import torch.nn as nn  # import PyTorch neural network
import torch.nn.functional as F  # import PyTorch neural network functional
from torch.autograd import Variable  # create variable from tensor
import torch.utils.data as Data  # create data from tensors
import uproot

#parameters that can be changed
place_to_access_and_save = "/Users/broad/Documents/Documents/university/summer work/DNN FOR PRESENTING/"
input_file = place_to_access_and_save + "callum_test.h5"
samples=["DNN_background", "DNN_signal", "data"]
model_structure = [128, 64, 32]
threshold = 0.5
variables=["delta_r_b","bl_invmass_b", "delta_eta_b", "bl_pt_b", "cluster_delta_r_b", "cluster_delta_eta_b", "mag_de_dr_b", "delta_r_bbar", "bl_invmass_bbar", "delta_eta_bar", "bl_pt_bbar", "cluster_delta_r_bbar", "cluster_delta_eta_bbar", "mag_de_dr_bbar","delta_angle_bbar", "delta_angle_b","bl_energy_b","bl_energy_bbar"]

file = h5py.File(place_to_access_and_save+'callum_test.h5', 'r')

# Access the dataset
number_jets = np.array(file['atlas/njets'])
met = np.array(file["atlas/MET"])
eventnumber = np.array(file["atlas/eventNumber"])
jets = np.array(file["atlas/jets"])
leptons = np.array(file["atlas/leptons"])
neutrinos = np.array(file["atlas/neutrinos"])

# %%
lepton_pt = leptons['pt']
lepton_eta = leptons['eta']
lepton_phi = leptons['phi']
lepton_energy = leptons['energy']
lepton_charge = leptons['charge']
lepton_type = leptons['type']
jets_pt = jets['pt']
jets_eta = jets['eta']
jets_phi = jets['phi']
jets_energy = jets['energy']
jets_isbtagged = jets["is_tagged"]

# %%
two_b_jets_mask = np.sum(jets_isbtagged,axis=1)==2
two_opposite_leptons_mask = lepton_charge[:,0]+lepton_charge[:,1] == 0
clear_up_mask = two_opposite_leptons_mask & two_b_jets_mask
jets_which_are_btagged = jets_isbtagged[clear_up_mask]
recon_j1=vector.zip({'pt': jets_pt[clear_up_mask][jets_which_are_btagged][::2], 'eta': jets_eta[clear_up_mask][jets_which_are_btagged][::2], 'phi': jets_phi[clear_up_mask][jets_which_are_btagged][::2], 'e': jets_energy[clear_up_mask][jets_which_are_btagged][::2]})
recon_j2=vector.zip({'pt': jets_pt[clear_up_mask][jets_which_are_btagged][1::2], 'eta': jets_eta[clear_up_mask][jets_which_are_btagged][1::2], 'phi': jets_phi[clear_up_mask][jets_which_are_btagged][1::2], 'e': jets_energy[clear_up_mask][jets_which_are_btagged][1::2]})
recon_leptons=vector.zip({'pt': lepton_pt, 'eta': lepton_eta, 'phi': lepton_phi, 'e': lepton_energy})[clear_up_mask]
lepton_positive_charge_mask = lepton_charge[clear_up_mask]==1
lepton_negative_charge_mask = lepton_charge[clear_up_mask]==-1
recon_lep_positive = recon_leptons[lepton_positive_charge_mask]
recon_lep_negative = recon_leptons[lepton_negative_charge_mask]

# %%
#detla R
signal_delta_r_jet1_positive = recon_lep_positive.deltaR(recon_j1)
signal_delta_r_jet2_negative = recon_lep_negative.deltaR(recon_j2)
#bl invariant mass
signal_invmass_jet1_positive = (recon_j1+recon_lep_positive).m
signal_invmass_jet2_negative = (recon_j2+recon_lep_negative).m
#delta eta 
signal_delta_eta_jet1_positive = recon_lep_positive.deltaeta(recon_j1)
signal_delta_eta_jet2_negative = recon_lep_negative.deltaeta(recon_j2)
#combined pt of bl
signal_combined_pt_jet1_positive = (recon_lep_positive+recon_j1).pt
signal_combined_pt_jet2_negative = (recon_lep_negative+recon_j2).pt
#delta_r between clusters
signal_cluster_delta_r_jet1_positive = (recon_lep_positive+recon_j1).deltaR(recon_lep_negative+recon_j2)
signal_cluster_delta_r_jet2_negative = (recon_lep_positive+recon_j1).deltaR(recon_lep_negative+recon_j2)
#delta_eta between clusters
signal_cluster_delta_eta_jet1_positive = (recon_lep_positive+recon_j1).deltaeta(recon_lep_negative+recon_j2)
signal_cluster_delta_eta_jet2_negative = (recon_lep_positive+recon_j1).deltaeta(recon_lep_negative+recon_j2)
#magnitude of delta_eta + delta_R
signal_mag_de_dr_jet1_positive = np.abs(signal_delta_eta_jet1_positive + signal_delta_r_jet1_positive)
signal_mag_de_dr_jet2_negative = np.abs(signal_delta_eta_jet2_negative + signal_delta_r_jet2_negative)

#bl energy
signal_energy_jet1_positive = (recon_j1+recon_lep_positive).e
signal_energy_jet2_negative = (recon_j2+recon_lep_negative).e

#delta angle
signal_delta_angle_jet1_positive = recon_lep_positive.deltaangle(recon_j1)
signal_delta_angle_jet2_negative = recon_lep_negative.deltaangle(recon_j2)

# %%
h5_file = h5py.File(place_to_access_and_save + "data.h5", 'w')
group = h5_file.create_group('Objects')
group.create_dataset("delta_r_b", data=signal_delta_r_jet1_positive[:])
group.create_dataset("bl_invmass_b", data=signal_invmass_jet1_positive[:])
group.create_dataset("delta_eta_b", data=signal_delta_eta_jet1_positive[:])
group.create_dataset("bl_pt_b", data=signal_combined_pt_jet1_positive[:])
group.create_dataset("cluster_delta_r_b", data=signal_cluster_delta_r_jet1_positive[:])
group.create_dataset("cluster_delta_eta_b", data=signal_cluster_delta_eta_jet1_positive[:])
group.create_dataset("mag_de_dr_b", data=signal_mag_de_dr_jet1_positive[:])
group.create_dataset("delta_r_bbar", data=signal_delta_r_jet2_negative[:])
group.create_dataset("bl_invmass_bbar", data=signal_invmass_jet2_negative[:])
group.create_dataset("delta_eta_bar", data=signal_delta_eta_jet2_negative[:])
group.create_dataset("bl_pt_bbar", data=signal_combined_pt_jet2_negative[:])
group.create_dataset("cluster_delta_r_bbar", data=signal_cluster_delta_r_jet2_negative[:])
group.create_dataset("cluster_delta_eta_bbar", data=signal_cluster_delta_eta_jet2_negative[:])
group.create_dataset("mag_de_dr_bbar", data=signal_mag_de_dr_jet2_negative[:])
group.create_dataset("delta_angle_b", data=signal_delta_angle_jet1_positive[:])
group.create_dataset("delta_angle_bbar", data=signal_delta_angle_jet2_negative[:])
group.create_dataset("bl_energy_b", data=signal_energy_jet1_positive[:])
group.create_dataset("bl_energy_bbar", data=signal_energy_jet2_negative[:])
h5_file.close()

#Now use the DNN

seed_value = 420  # 42 is the answer to life, the universe and everything
from numpy.random import seed  # import the function to set the random seed in NumPy
seed(seed_value)  # set the seed value for random numbers in NumPy

DataFrames = {}
for s in samples:
    h5_file = h5py.File(place_to_access_and_save + s + ".h5", 'r')
    group = h5_file['Objects']
    data_dict = {}
    
    for v in variables:
        data = group[v][:]
        data_dict[v] = data

    df = pd.DataFrame(data_dict)
    df.insert(0, 'entry', range(len(df)))
    DataFrames[s] = df
    
    h5_file.close()

ML_inputs = variables
all_MC = []  # define empty list that will contain all features for the MC
for s in samples:  # loop over the different samples
    if s != "data":  # only MC should pass this
        all_MC.append(
            DataFrames[s][ML_inputs]
        )  # append the MC dataframe to the list containing all MC features
X = np.concatenate(
    all_MC
)  # concatenate the list of MC dataframes into a single 2D array of features, called X


from sklearn.model_selection import train_test_split

# make train and test sets
X_train, _,_,_ = train_test_split(
    X, np.ones(len(X)), test_size=0.2, random_state=seed_value
)  # set the random seed for reproducibility
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # initialise StandardScaler

# Fit only to the training data
scaler.fit(X_train)


class Classifier_MLP(nn.Module):  # define Multi-Layer Perceptron
    def __init__(self, in_dim, hidden_dims, out_dim):  # initialise
        super().__init__()  # lets you avoid referring to the base class explicitly
        
        layers = []
        
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.01))
            prev_dim = hidden_dim
            
        layers.append(torch.nn.Linear(prev_dim, out_dim))
        
        self.model = nn.Sequential(*layers)
        
        self.out_dim = out_dim

    def forward(self, x):  # order of the layers
        logits = self.model(x)
        probabilities = nn.functional.softmax(logits, dim=1)
        
        return logits, probabilities #Return both logits and probabilities


input_size = len(ML_inputs)  # The number of features
print(len(ML_inputs))
hidden_size=model_structure
print(model_structure)
num_classes = 2  # The number of output classes. In this case: [signal, background]
torch.manual_seed(seed_value)  # set random seed for PyTorch


NN_clf = Classifier_MLP(in_dim=input_size, hidden_dims=hidden_size, out_dim=num_classes)
state_dict_path = place_to_access_and_save+ f'/model {str(hidden_size)}.pth'
NN_clf.load_state_dict(torch.load(state_dict_path), strict=False)



X_data = DataFrames['data'][ML_inputs].values # .values converts straight to NumPy array
X_data_scaled = scaler.transform(X_data) # X_data now scaled same as training and testing sets

X_data_tensor = torch.as_tensor(
    X_data_scaled, dtype=torch.float
)  # make tensor from X_test_scaled


X_data_var = Variable(X_data_tensor)
# make variables from tensors

# Get the probabilities from the model output
NN_clf.eval()
_, prob = NN_clf(X_data_var)
probabilities = prob.cpu().detach().numpy()

signal_probabilities = probabilities[:, 1]

# Apply the new threshold to determine the class labels
results = (signal_probabilities >= threshold).astype(int).astype(bool)

#Now make sense of data passed out of neural network
result_b = np.where(results, recon_j1, recon_j2)
result_bbar = np.where(results, recon_j2, recon_j1)
result_positive_lep = recon_lep_positive
result_negative_lep = recon_lep_negative



#save results to h5file
results_file = h5py.File(place_to_access_and_save + "results.h5", 'w')
b = results_file.create_group('bjet')
b.create_dataset("pt", data=result_b.pt)
b.create_dataset("eta", data=result_b.eta)
b.create_dataset("phi", data=result_b.phi)
b.create_dataset("m", data=result_b.m)

bbar = results_file.create_group('bbarjet')
bbar.create_dataset("pt", data=result_bbar.pt)
bbar.create_dataset("eta", data=result_bbar.eta)
bbar.create_dataset("phi", data=result_bbar.phi)
bbar.create_dataset("m", data=result_bbar.m)

positive_lep = results_file.create_group('positive_lep')
positive_lep.create_dataset("pt", data=result_positive_lep.pt)
positive_lep.create_dataset("eta", data=result_positive_lep.eta)
positive_lep.create_dataset("phi", data=result_positive_lep.phi)
positive_lep.create_dataset("m", data=result_positive_lep.m)

negative_lep = results_file.create_group('negative_lep')
negative_lep.create_dataset("pt", data=result_negative_lep.pt)
negative_lep.create_dataset("eta", data=result_negative_lep.eta)
negative_lep.create_dataset("phi", data=result_negative_lep.phi)
negative_lep.create_dataset("m", data=result_negative_lep.m)

event = results_file.create_group("event")
event.create_dataset("eventnumber", data=(eventnumber[clear_up_mask]).flatten())

results_file.close()

print("done, results saved to results.txt folder"
      )

print(eventnumber[clear_up_mask])

print(len((eventnumber[clear_up_mask]).flatten()))