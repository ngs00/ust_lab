import pandas
import numpy
import itertools
from tqdm import tqdm
from ust_lab.util.chem import get_form_vec


def load_form_vec_dataset(path_dataset, elem_attrs, idx_form, idx_target):
    data = pandas.read_excel(path_dataset).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(data))):
        form_vec = get_form_vec(data[i][idx_form], elem_attrs)
        dataset.append(numpy.hstack([form_vec, data[i][idx_target]]))

    return numpy.vstack(dataset)


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
