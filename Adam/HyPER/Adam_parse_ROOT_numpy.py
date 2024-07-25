import h5py
import uproot
import argparse
import numpy as np
from tqdm import tqdm
import awkward as ak


def pad(variable, length, value=0):
    padded = ak.pad_none(variable, length, axis=1, clip = True)
    return np.array(ak.fill_none(padded, value))


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


    # Open HDF5 file for chunked writing
    h5_file = h5py.File(output, 'w')
    inputs_group = h5_file.create_group('INPUTS')
    labels_group = h5_file.create_group('LABELS')
    inputs_jet = inputs_group.create_dataset("jet", shape=(num_entries, pad_to_jet, 6), dtype=np.float32, chunks=(chunk_size, pad_to_jet, 6))
    labels_VertexID = labels_group.create_dataset("VertexID", shape=(num_entries, pad_to_jet), dtype=np.int32, chunks=(chunk_size, pad_to_jet))
    inputs_global = inputs_group.create_dataset("global", shape=(num_entries, 2), dtype=np.float32, chunks=(chunk_size, 2))
    labels_IndexSelect = labels_group.create_dataset("IndexSelect", shape=(num_entries,), dtype=np.int32, chunks=(chunk_size,))


    values = ['jet_bTag','jet_eta','jet_phi','jet_e','jet_pt','jet_truthmatch','njet','nbTagged','allMatchedEvent']
    for chunk, array in tqdm(enumerate(uproot.iterate(input, values, step_size=chunk_size))):
            jet_bTag = pad(array["jet_bTag"], pad_to_jet)
            jet_eta = pad(array["jet_eta"], pad_to_jet)
            jet_phi = pad(array["jet_phi"], pad_to_jet)
            jet_e = pad(array["jet_e"], pad_to_jet)
            jet_pt = pad(array["jet_pt"], pad_to_jet)
            jet_match = array["jet_truthmatch"]
            njet = array["njet"]
            nbTagged = array["nbTagged"]
            allmatched = array["allMatchedEvent"]
            id = np.ones((len(njet), pad_to_jet))


            jet_data = np.transpose(np.stack((jet_e, jet_eta, jet_phi, jet_pt, jet_bTag, id)), (1,2,0))
            VertexID_data = pad(jet_match, pad_to_jet, -9)
            global_data = np.stack((njet, nbTagged), axis=1)

            start_idx = chunk * chunk_size
            end_idx = start_idx+len(njet)

            inputs_jet[start_idx:end_idx] = jet_data
            labels_VertexID[start_idx:end_idx] = VertexID_data
            inputs_global[start_idx:end_idx] = global_data
            labels_IndexSelect[start_idx:end_idx] = allmatched

    h5_file.close()
    root_file.close()





if __name__ == "__main__":
    args = argparser()
    MakeDataset(args.input, args.output)

