import pandas
import numpy
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from ust_lab.util.chem import get_form_vec, get_mol_graph


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


def get_tensor_dataset_loader(dataset, batch_size=128, shuffle=False):
    dataset_x = torch.tensor(dataset[:, :-1], dtype=torch.float)
    dataset_y = torch.tensor(dataset[:, -1], dtype=torch.float).view(-1, 1)
    
    return DataLoader(TensorDataset(dataset_x, dataset_y), batch_size=batch_size, shuffle=shuffle)
