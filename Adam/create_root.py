import sys
import pylhe
import numpy as np
import uproot
import sys


def calculate(awkward_array):
    p = ['d','u','s','c','b','t',"b'","t'",'g','el','el_vu','mu','mu_vu','tau','tau_vu', "tau'", "tau'_nu", 'g','a','Z','W']
    particles = {}
    addon = 1
    for i,part in enumerate(p):
        if (i+addon) in [10,19,20]: #skip these pgids
            addon+=1
        particles[i+addon] = part
        if (i+addon) <=8:
            particles[-1*(i+addon)] = part+'bar' #for quarks, give a bar option

    data = {}
    charge = {}
    for key, value in particles.items():
        if key >= 11:
            temp = awkward_array.vector[np.abs(awkward_array.id) == key]
            if np.sum(temp.pt) >0:
                data[value] = temp #if it is a lepton, want both particles and anti particles
                charge[value] = -1*np.sign(awkward_array.id)[np.abs(awkward_array.id) == key] #also want the sign of this selection
        else:
            temp =  awkward_array.vector[awkward_array.id == key]
            if np.sum(temp.pt) >0:
                data[value] = temp


    return data, charge


file_name = sys.argv[1]
outfile  = sys.argv[2]
awkward_array =pylhe.to_awkward(pylhe.read_lhe_with_attributes(file_name)).particles
data, charge = calculate(awkward_array)
with uproot.recreate(outfile) as f:
    tree = {}
    for key, value in data.items():
        indices = np.argsort(-value.pt, axis=1)
        tree[f'{key}_pt'] = value.pt[indices]
        tree[f'{key}_eta'] = value.eta[indices]
        tree[f'{key}_phi'] = value.phi[indices]
        tree[f'{key}_e'] = value.e[indices]
        if key in charge:
            tree[f'{key}_charge'] = charge[key][indices]
    f["tree"] = tree
