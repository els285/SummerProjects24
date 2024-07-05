# %%
#run in conda environment PYTHON 3.11.7
#Need background_ATLAS.h5 and signal_ATLAS.h5 inside the file in order to get correct scaling as this was used for the training
#need data.h5 file as well although code here can be easily modified to exclude this
#need the saved neural network model in the file

import uproot
import vector
import pandas as pd  # to store data as dataframe
import numpy as np  # for numerical calculations such as histogramming
import matplotlib.pyplot as plt  # for plotting
import h5py
import torch  # import PyTorch
import torch.nn as nn  # import PyTorch neural network
import torch.nn.functional as F  # import PyTorch neural network functional
from torch.autograd import Variable  # create variable from tensor
import torch.utils.data as Data  # create data from tensors

# %%

seed_value = 420  # 42 is the answer to life, the universe and everything
from numpy.random import seed  # import the function to set the random seed in NumPy
seed(seed_value)  # set the seed value for random numbers in NumPy

place_to_access_and_save = "/Users/broad/OneDrive/Documents/university/summer work/project 2/"
samples=["background_ATLAS", "signal_ATLAS", "data"]
variables=["delta_r", "bl_invmass", "delta_eta", "bl_pt", "cluster_delta_r", "cluster_delta_eta", "mag_de_dr"]
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

# %%
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

# %%
from sklearn.model_selection import train_test_split

# make train and test sets
X_train, _,_,_ = train_test_split(
    X, np.ones(len(X)), test_size=0.2, random_state=seed_value
)  # set the random seed for reproducibility
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # initialise StandardScaler

# Fit only to the training data
scaler.fit(X_train)

# %%
class Classifier_MLP(nn.Module):  # define Multi-Layer Perceptron
    def __init__(self, in_dim, hidden_dims, out_dim):  # initialise
        super().__init__()  # lets you avoid referring to the base class explicitly
        
        layers = []
        
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(torch.nn.Linear(prev_dim, out_dim))
        
        self.model = nn.Sequential(*layers)
        
        self.out_dim = out_dim

    def forward(self, x):  # order of the layers
        logits = self.model(x)
        probabilities = nn.functional.softmax(logits, dim=1)
        
        return logits, probabilities #Return both logits and probabilities

# %%
input_size = len(ML_inputs)  # The number of features
num_classes = 2  # The number of output classes. In this case: [signal, background]
hidden_size = [32,16,8] # The number of nodes at the hidden layer
torch.manual_seed(seed_value)  # set random seed for PyTorch

# %%
NN_clf = Classifier_MLP(
    in_dim=input_size, hidden_dims=hidden_size, out_dim=num_classes
)  # call Classifier_MLP class
saved_model_path = place_to_access_and_save + f'model {str(hidden_size)}.pth'
NN_clf.load_state_dict(torch.load(saved_model_path))


# %%
X_data = DataFrames['data'][ML_inputs].values # .values converts straight to NumPy array
X_data_scaled = scaler.transform(X_data) # X_data now scaled same as training and testing sets

X_data_tensor = torch.as_tensor(
    X_data_scaled, dtype=torch.float
)  # make tensor from X_test_scaled


X_data_var = Variable(X_data_tensor)
# make variables from tensors

out, prob = NN_clf(X_data_var)
results = prob.cpu().detach().numpy().argmax(axis=1)

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Generate the ground truth labels
expected_labels = np.concatenate([np.ones(22762), np.zeros(22762)])

results = prob.cpu().detach().numpy().argmax(axis=1)

np.savetxt("results", results, delimiter=',')

print("results saved" )


