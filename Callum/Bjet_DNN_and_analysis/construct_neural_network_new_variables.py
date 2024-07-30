# %%
import pandas as pd  # to store data as dataframe
import numpy as np  # for numerical calculations such as histogramming
import matplotlib.pyplot as plt  # for plotting
import h5py

# %%
seed_value = 420  # 42 is the answer to life, the universe and everything
from numpy.random import seed  # import the function to set the random seed in NumPy

seed(seed_value)  # set the seed value for random numbers in NumPy


#read files
place_to_access_and_save = "/Users/broad/Documents/Documents/university/summer work/DNN FOR PRESENTING/"
samples=["DNN_signal", "DNN_background"]
variables=["delta_r_b","bl_invmass_b", "delta_eta_b", "bl_pt_b", "cluster_delta_r_b", "cluster_delta_eta_b", "mag_de_dr_b", "delta_r_bbar", "bl_invmass_bbar", "delta_eta_bar", "bl_pt_bbar", "cluster_delta_r_bbar", "cluster_delta_eta_bbar", "mag_de_dr_bbar","delta_angle_bbar", "delta_angle_b","bl_energy_b","bl_energy_bbar"]
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

delta_r = {  # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    "bin_width": 1,  # width of each histogram bin
    "num_bins": 10,  # number of histogram bins
    "xrange_min": 6,  # minimum on x-axis
    "xlabel": r"$\delta r",  # x-axis label
}
lb_invmass = {  # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    "bin_width": 1,  # width of each histogram bin
    "num_bins": 28,  # number of histogram bins
    "xrange_min": 10,  # minimum on x-axis
    "xlabel": r"$M\_lb$ [GeV]",  # x-axis label
}

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
all_y = (
    []
)  # define empty list that will contain labels whether an event in signal or background
for s in samples:  # loop over the different samples
    if s != "data":  # only MC should pass this
        if "signal" in s:  # only signal MC should pass this
            all_y.append(
                np.ones(DataFrames[s].shape[0])
            )  # signal events are labelled with 1
        else:  # only background MC should pass this
            all_y.append(
                np.zeros(DataFrames[s].shape[0])
            )  # background events are labelled 0
y = np.concatenate(
    all_y
) 


# %%
from sklearn.model_selection import train_test_split

# make train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed_value
)  # set the random seed for reproducibility
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # initialise StandardScaler

# Fit only to the training data
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)


X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# %%
import torch  # import PyTorch
import torch.nn as nn  # import PyTorch neural network
import torch.nn.functional as F  # import PyTorch neural network functional
from torch.autograd import Variable  # create variable from tensor
import torch.utils.data as Data  # create data from tensors

# %%
# Parameters
epochs = 30 # number of training epochs
batch_size = 64  # number of samples per batch
input_size = len(ML_inputs)  # The number of features
num_classes = 2  # The number of output classes. In this case: [signal, background]
hidden_size = [128,64,32] # The number of nodes at the hidden layer
learning_rate = 0.001  # The speed of convergence
verbose = True  # flag for printing out stats at each epoch
torch.manual_seed(seed_value)  # set random seed for PyTorch

with open(place_to_access_and_save + "output.txt", 'w') as file:
    file.write("========================begin==============================\n")

# %%
X_train_tensor = torch.as_tensor(
    X_train_scaled, dtype=torch.float
)  # make tensor from X_train_scaled
y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)  # make tensor from y_train

X_train_var, y_train_var = Variable(X_train_tensor), Variable(
    y_train_tensor
)  # make variables from tensors

X_tensor = torch.as_tensor(
    X_scaled, dtype=torch.float
)  # make tensor from X_train_scaled
y_tensor = torch.as_tensor(y, dtype=torch.long)

X_var, y_var = Variable(X_tensor), Variable(y_tensor)



X_valid_var, y_valid_var = (
    X_var,
    y_var,
)  # get first 100 events for validation
X_train_nn_var, y_train_nn_var = (
    X_var,
    y_var,
)  # get remaining events for training

train_data = Data.TensorDataset(
    X_train_nn_var, y_train_nn_var
)  # create training dataset
valid_data = Data.TensorDataset(X_valid_var, y_valid_var)  # create validation dataset

train_loader = Data.DataLoader(
    dataset=train_data,  # PyTorch Dataset
    batch_size=batch_size,  # how many samples per batch to load
    shuffle=True,
)  # data reshuffled at every epoch

valid_loader = Data.DataLoader(
    dataset=valid_data,  # PyTorch Dataset
    batch_size=batch_size,  # how many samples per batch to load
    shuffle=True,
)  # data reshuffled at every epoch

# %%
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

# %%
NN_clf = Classifier_MLP(
    in_dim=input_size, hidden_dims=hidden_size, out_dim=num_classes
)  # call Classifier_MLP class
optimizer = torch.optim.Adam(
    NN_clf.parameters(), lr=learning_rate
)  # optimize model parameters



# %%
_results = []  # define empty list for epoch, train_loss, valid_loss, accuracy
for epoch in range(epochs):  # loop over the dataset multiple times

    # training loop for this epoch
    NN_clf.train()  # set the model into training mode

    train_loss = 0.0  # start training loss counter at 0
    for batch, (x_train_batch, y_train_batch) in enumerate(
        train_loader
    ):  # loop over train_loader

        NN_clf.zero_grad()  # set the gradients to zero before backpropragation because PyTorch accumulates the gradients
        out, prob = NN_clf(
            x_train_batch
        )  # get output and probability on this training batch
        loss = F.cross_entropy(out, y_train_batch)  # calculate loss as cross entropy

        loss.backward()  # compute dloss/dx
        optimizer.step()  # updates the parameters

        train_loss += loss.item() * x_train_batch.size(
            0
        )  # add to counter for training loss

    train_loss /= len(
        train_loader.dataset
    )  # divide train loss by length of train_loader

    if verbose:  # if verbose flag set to True
        with open(place_to_access_and_save + "output.txt", 'a') as file:
            file.write("Epoch: {}, Train Loss: {:4f}, \n".format(epoch, train_loss))
        print("Epoch: {}, Train Loss: {:4f}".format(epoch, train_loss))

    # validation loop for this epoch:
    NN_clf.eval()  # set the model into evaluation mode
    with torch.no_grad():  # turn off the gradient calculations

        correct = 0
        valid_loss = 0  # start counters for number of correct and validation loss
        for i, (x_valid_batch, y_valid_batch) in enumerate(
            valid_loader
        ):  # loop over validation loader

            out, prob = NN_clf(
                x_valid_batch
            )  # get output and probability on this validation batch
            loss = F.cross_entropy(out, y_valid_batch)  # compute loss as cross entropy

            valid_loss += loss.item() * x_valid_batch.size(
                0
            )  # add to counter for validation loss

            preds = prob.argmax(dim=1, keepdim=True)  # get predictions
            correct += (
                preds.eq(y_valid_batch.view_as(preds)).sum().item()
            )  # count number of correct

        valid_loss /= len(
            valid_loader.dataset
        )  # divide validation loss by length of validation dataset
        accuracy = correct / len(
            valid_loader.dataset
        )  # calculate accuracy as number of correct divided by total

    if verbose:  # if verbose flag set to True
        print(
            "Validation Loss: {:4f}, Validation Accuracy: {:4f}".format(
                valid_loss, accuracy
            )
        )
        with open(place_to_access_and_save + "output.txt", 'a') as file:
            file.write("Validation Loss: {:4f}, Validation Accuracy: {:4f}\n".format(
                valid_loss, accuracy)
            )
    # create output row:
    _results.append([epoch, train_loss, valid_loss, accuracy])

results = np.array(_results)  # make array of results
print("Finished Training")
print("Final validation error: ", 100.0 * (1 - accuracy), "%")
with open(place_to_access_and_save + "output.txt", 'a') as file:
    file.write("Finished Training\n")
    string = f"Final validation accuracy: {100.0 * accuracy:.2f}%"
    file.write(string)

# %%
# Extracting data for plotting
epochs = results[:, 0]
train_losses = results[:, 1]
valid_losses = results[:, 2]
accuracies = results[:, 3]

# Plotting the loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.plot(epochs, valid_losses, label='Validation Loss', marker='s')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Define the name of the directory to be created
import os
directory_name = (place_to_access_and_save + "layers_" + str(hidden_size) + "learning_rate_" + str(learning_rate))
os.mkdir(directory_name)
loss_plot = os.path.join(directory_name, 'loss')
plt.savefig(loss_plot)
plt.close()


# Plotting the accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracies, label='Validation Accuracy', marker='o', color='r')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
accuracy_plot = os.path.join(directory_name, "accuracy")
plt.savefig(accuracy_plot)
plt.close()



# %%

X_test_tensor = torch.as_tensor(
    X_test_scaled, dtype=torch.float
)  # make tensor from X_test_scaled
y_test_tensor = torch.as_tensor(y_test, dtype=torch.long)  # make tensor from y_test

X_test_var, y_test_var = Variable(X_test_tensor), Variable(
    y_test_tensor
)  # make variables from tensors

out, prob = NN_clf(X_test_var)  # get output and probabilities from X_test
y_pred_NN = (
    prob.cpu().detach().numpy().argmax(axis=1)
)  # get signal/background predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_NN))




plt.close()
from sklearn.metrics import classification_report, roc_auc_score
import numpy
decisions_nn = (
    NN_clf(X_test_var)[1][:, 1].cpu().detach().numpy()
)  # get the decisions of the neural network

# %%
from sklearn.metrics import roc_curve
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, decisions_nn) # get FPRs, TPRs and thresholds for neural network

plt.plot(
    fpr_nn, tpr_nn, linestyle="dashed", label="Neural Network"
)  # plot neural network ROC
plt.plot(
    [0, 1], [0, 1], linestyle="dotted", color="grey", label="Luck"
)  # plot diagonal line to indicate luck
plt.xlabel("False Positive Rate")  # x-axis label
plt.ylabel("True Positive Rate")  # y-axis label
plt.grid()  # add a grid to the plot
plt.legend()  # add a legend
plot_filename = os.path.join(directory_name, 'positve_rate')
plt.savefig(plot_filename)
plt.close()


plt.hist(
    decisions_nn[y_test == 0], histtype="step", bins=50, label="Background Events"
)  # plot background
plt.hist(
    decisions_nn[y_test == 1],
    histtype="step",
    bins=50,
    linestyle="dashed",
    label="Signal Events",
)  # plot signal
plt.xlabel("Threshold")  # x-axis label
plt.ylabel("Number of Events")  # y-axis label
plt.semilogy()  # make the y-axis semi-log
plt.legend()  # draw the legend
plot_filename = os.path.join(directory_name, 'cuts')
plt.savefig(plot_filename)
plt.close()
#save model
torch.save(NN_clf.state_dict(), directory_name + f'/model {str(hidden_size)}.pth')


