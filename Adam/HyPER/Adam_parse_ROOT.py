import h5py
import uproot
import argparse
import numpy as np
from tqdm import tqdm



def argparser():
    parser = argparse.ArgumentParser(description='Make a HyPER dataset')
    parser.add_argument('-f', '--input',  type=str, required=True, help='ROOT input file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output .h5 file')
    return parser.parse_args()


def MakeDataset(input: str, output: str):
    r"""A example for constructing `HyPER` dataset.

    Args:
        input (str): input ROOT file.
        output (str): output HyPER dataset in .h5 format.
    """
    # Max number of objects (padding)
    pad_to_jet = 20

    # Load the ROOT file
    root_file = uproot.open(input)
    tree = root_file['Delphes']
    num_entries = tree.num_entries
    root_file.close()
    chunk_size = 200000  # Define your chunk size based on memory limits

    node_dt = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.float32), ('id', np.float32)])
    global_dt = np.dtype([('njet', np.float32), ('nbTagged', np.float32)])

    # Open HDF5 file for chunked writing
    h5_file = h5py.File(output, 'w')
    inputs_group = h5_file.create_group('INPUTS')
    labels_group = h5_file.create_group('LABELS')
    inputs_jet = inputs_group.create_dataset("jet", shape=(num_entries, pad_to_jet), dtype=node_dt, chunks=(chunk_size, pad_to_jet))
    labels_VertexID = labels_group.create_dataset("VertexID", shape=(num_entries, pad_to_jet), dtype=np.int32, chunks=(chunk_size, pad_to_jet))
    inputs_global = inputs_group.create_dataset("global", shape=(num_entries, 2), dtype=global_dt, chunks=(chunk_size, 2))
    labels_IndexSelect = labels_group.create_dataset("IndexSelect", shape=(num_entries,), dtype=np.int32, chunks=(chunk_size,))

    values = ['jet_pt','jet_eta','jet_phi','jet_e','jet_bTag','jet_truthmatch','njet','nbTagged']
    
    for chunk, array in tqdm(enumerate(uproot.iterate(f'{input}:Delphes', values, step_size=chunk_size, library='np'))):
        jet_bTag = array["jet_bTag"]
        jet_eta = array["jet_eta"]
        jet_phi = array["jet_phi"]
        jet_e = array["jet_e"]
        jet_pt = array["jet_pt"]
        jet_match = array["jet_truthmatch"]
        njet = array["njet"]
        nbTagged = array["nbTagged"]
        start_idx = chunk * chunk_size
        end_idx = start_idx+len(njet)
        jet_data = np.zeros((end_idx - start_idx, pad_to_jet), dtype=node_dt)
        global_data = np.zeros((end_idx - start_idx, 2), dtype=global_dt)
        VertexID_data = np.full((end_idx - start_idx, pad_to_jet), -9, dtype=np.int32)  # use -9 for none filled values
        IndexSelect_data = np.zeros(end_idx - start_idx, dtype=np.int32)
        for i in range(end_idx - start_idx):
            num_jets = njet[i]
            for j in range(num_jets):
                jet_data[i][j] = (jet_e[i][j], jet_eta[i][j], jet_phi[i][j], jet_pt[i][j], jet_bTag[i][j], 1)
            global_data[i] = [njet[i], nbTagged[i]]
            VertexID_data[i, :num_jets] = jet_match[i]
            IndexSelect_data[i] = 1 if len(set(jet_match[i]).intersection(set([1, 2, 3, 4, 5, 6]))) == 6 else 0
        
        inputs_jet[start_idx:end_idx] = jet_data
        labels_VertexID[start_idx:end_idx] = VertexID_data
        inputs_global[start_idx:end_idx] = global_data
        labels_IndexSelect[start_idx:end_idx] = IndexSelect_data

    h5_file.close()
    root_file.close()




if __name__ == "__main__":
    args = argparser()
    MakeDataset(args.input, args.output)

