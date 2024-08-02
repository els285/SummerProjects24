import numpy as np
from tqdm import tqdm
import h5py
import sys
import os


file_names = [f'ttbar_allhad_input_{i}.h5' for i in range(50)]
#file_names = ['skim2_28.h5']
p_status_len = 6
pad_to_jet = 20
test_val_size = 0.1

# path = '/mnt/c/Users/HP/Documents/data/HyPER'
# os.chdir(path)
train_file  = sys.argv[1]
if len(sys.argv)==2:
    train_size = 1
elif len(sys.argv)==3:
    test_file = sys.argv[2]
    train_size = 1-test_val_size
elif len(sys.argv) ==4:
    test_file = sys.argv[2]
    val_file = sys.argv[3]
    train_size = 1-2*test_val_size
else:
    print('input must be in the form train.h5 test.h5(optional) val.h5(optional)')
    quit()

def read_h5(filename):
    dic = {}
    with h5py.File(filename, 'r') as h5_file:
        group = h5_file['INPUTS']
        sub_dic = {}
        for name in group.keys():
            data = np.array(group[name])
            sub_dic[name] = data
        dic['INPUTS'] = sub_dic
        group = h5_file['LABELS']
        sub_dic = {}
        for name in group.keys():
            data = np.array(group[name])
            sub_dic[name] = data
        dic['LABELS'] = sub_dic
    return dic

def pad_array(array, pad_to_jet, dtype = np.float32):
    prev_len = len(array[0])
    padded_arr = np.zeros((len(array), pad_to_jet), dtype=dtype)
    padded_arr[:,:min(prev_len,pad_to_jet)] = array[:,:min(prev_len,pad_to_jet)]
    return padded_arr


global_dt   = np.dtype([('njet', np.float32), ('nbTagged', np.float32)])
node_dt  = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.float32), ('charge', np.float32), ('id', np.float32)])

jet_data = np.zeros((0, pad_to_jet), dtype=node_dt)
global_data = np.zeros((0, 1), dtype=global_dt)
VertexID_data = np.zeros((0, pad_to_jet), dtype=np.int64)
IndexSelect_data = np.zeros((0), dtype=np.int32)


for file in file_names:
    print(f'adding data from {file}')
    data = read_h5(file)
    jet_data = np.vstack((jet_data, pad_array(data['INPUTS']['jet'], pad_to_jet, node_dt)))
    global_data = np.vstack((global_data, data['INPUTS']['global']))
    VertexID_data = np.vstack((VertexID_data, pad_array(data['LABELS']['VertexID'], pad_to_jet)))
    IndexSelect_data = np.hstack((IndexSelect_data, data['LABELS']['IndexSelect']))
    


indices = np.arange(len(jet_data))
np.random.shuffle(indices)
jet_data = jet_data[indices]
global_data = global_data[indices]
VertexID_data = VertexID_data[indices]
train_num = int(train_size*len(jet_data))

print(f'saving train data to {train_file} ({train_num} events)')
h5_file = h5py.File(train_file, 'w')
inputs_group = h5_file.create_group('INPUTS')
labels_group = h5_file.create_group('LABELS')

inputs_group.create_dataset("jet", data=jet_data[:train_num])
inputs_group.create_dataset("global", data=global_data[:train_num])
labels_group.create_dataset("VertexID", data=VertexID_data[:train_num])
labels_group.create_dataset("IndexSelect", data = IndexSelect_data[:train_num])
h5_file.close()

if len(sys.argv) >=3:
    test_val_num = int(np.ceil(test_val_size*len(jet_data)))
    print(f'saving test data to {test_file} ({test_val_num} events)')
    h5_file = h5py.File(test_file, 'w')
    inputs_group = h5_file.create_group('INPUTS')
    labels_group = h5_file.create_group('LABELS')

    inputs_group.create_dataset("jet", data=jet_data[train_num:train_num+test_val_num])
    inputs_group.create_dataset("global", data=global_data[train_num:train_num+test_val_num])
    labels_group.create_dataset("VertexID", data=VertexID_data[train_num:train_num+test_val_num])
    labels_group.create_dataset("IndexSelect", data = IndexSelect_data[train_num:train_num+test_val_num])
    h5_file.close()
    
    if len(sys.argv) == 4:
        print(f'saving val data to {val_file} ({len(jet_data)-(train_num+test_val_num)} events)')
        h5_file = h5py.File(test_file, 'w')
        inputs_group = h5_file.create_group('INPUTS')
        labels_group = h5_file.create_group('LABELS')

        inputs_group.create_dataset("jet", data=jet_data[train_num+test_val_num:])
        inputs_group.create_dataset("global", data=global_data[train_num+test_val_num:])
        labels_group.create_dataset("VertexID", data=VertexID_data[train_num+test_val_num:])
        labels_group.create_dataset("IndexSelect", data = IndexSelect_data[train_num+test_val_num:])
        h5_file.close()

print('program finished')
