import pandas
import numpy
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
from rdkit import Chem
from pymatgen.core import Structure
from ust_lab.util.chem import get_form_vec, get_mol_graph, get_crystal_graph


def load_form_vec_dataset(path_dataset, elem_attrs, idx_form, idx_target):
    data = pandas.read_excel(path_dataset).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(data))):
        form_vec = get_form_vec(data[i][idx_form], elem_attrs)
        dataset.append(numpy.hstack([form_vec, data[i][idx_target]]))

    return numpy.vstack(dataset)


def load_mol_dataset(path_dataset, elem_attrs, idx_smiles, idx_target):
    data = pandas.read_excel(path_dataset).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(data))):
        mol = Chem.MolFromSmiles(data[i][idx_smiles])
        mol_graph = get_mol_graph(mol, elem_attrs, data[i][idx_target])

        if mol_graph is not None:
            dataset.append(mol_graph)

    return dataset


def load_mat_dataset(path_metadata, path_structs, elem_attrs, idx_mat_id, idx_target, atomic_cutoff=4.0):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=64)
    dataset = list()

    for i in tqdm(range(0, len(metadata))):
        mat = Structure.from_file('{}/{}.cif'.format(path_structs, metadata[i][idx_mat_id]))
        crystal_graph = get_crystal_graph(mat, elem_attrs, rbf_means, metadata[i][idx_target], atomic_cutoff)

        if crystal_graph is not None:
            dataset.append(crystal_graph)
    
    return dataset
    

def get_k_folds(dataset, n_folds, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    k_folds = list()
    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), n_folds)

    for i in range(0, n_folds):
        idx_train = list(itertools.chain.from_iterable(idx_rand[:i] + idx_rand[i+1:]))
        idx_test = idx_rand[i]

        if isinstance(dataset, list):
            dataset_train = [dataset[idx] for idx in idx_train]
            dataset_test = [dataset[idx] for idx in idx_test]
        else:
            dataset_train = dataset[idx_train]
            dataset_test = dataset[idx_test]
        k_folds.append([dataset_train, dataset_test])

    return k_folds


def get_tensor_dataset(dataset):
    x = torch.tensor(dataset[:, :-1], dtype=torch.float)
    y = torch.tensor(dataset[:, -1], dtype=torch.float).view(-1, 1)

    return TensorDataset(x, y)
