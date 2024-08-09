import pylhe
import numpy as np
import uproot
import gzip
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





def main():
    # Check if enough arguments are provided
    if len(sys.argv) not in  [2,3]:
        print("Usage: python program.py <folder_name> <run_number ##>(optional)")
        return

    # Get file names from command line arguments
    if len(sys.argv) == 3:
        run = sys.argv[2]
    else:
        run = '01'

    file_name = f'{sys.argv[1]}/Events/run_{run}/unweighted_events.lhe'
    path = '/mnt/c/Users/HP/Documents/data/EFT_new/'
    name = sys.argv[1]
    if run != '01':
        name +='_'+run
    destination_name = path+name+'.root'
    try:
        # Open the source file for reading

        with gzip.open(file_name+'.gz', 'rb') as gz_file:
            with open(file_name, 'wb') as out_file:
                out_file.write(gz_file.read())

        awkward_array =pylhe.to_awkward(pylhe.read_lhe_with_attributes(file_name))
        data, charge = calculate(awkward_array.particles)
        with uproot.recreate(destination_name) as f:
            tree = {}
            for key, value in data.items():
                indices = np.argsort(-value.pt, axis=1)
                tree[f'{key}_pt'] = value.pt[indices]
                tree[f'{key}_eta'] = value.eta[indices]
                tree[f'{key}_phi'] = value.phi[indices]
                tree[f'{key}_e'] = value.e[indices]
                tree['weights'] = awkward_array.eventinfo.weight
                if key in charge:
                    tree[f'{key}_charge'] = charge[key][indices]
            f["tree"] = tree

        parameters = {}
        vars = ['cQQ1', 'cQQ8' ,'cQt1', 'cQt8' ,'ctt1', 'ctG']
        with open(file_name, 'r') as file:
            for raw in file:
                line = raw.strip().split(' ')
                if line[0] == 'generate':
                    parameters['generate'] = ' '.join(line[1:])
                elif 'nevents' in line:
                    parameters['nevents'] = line[0]
                elif 'Integrated weight' in raw:
                    parameters['cross_section'] = raw.split(':')[-1].strip()
                elif vars:
                    var = [var for var in vars if var in line]
                    if var:
                        parameters[var[0]] = line[1]
                        vars.remove(var[0])
                elif parameters.get('cross_section') and not vars:
                    break
        with open(path + 'run_info.txt', 'a') as f:
            string = '\n'.join([f'{key} : {value}' for key, value in parameters.items()]) + '\n' + '-'*40 + '\n'
            f.write(name+'\n'+string)


        print(f"Successfully copied content from {file_name} to {destination_name}")

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for file '{file_name}' or '{destination_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
