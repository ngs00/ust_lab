import pandas
import json
import math
import numpy
import itertools
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
from chemparse import parse_formula
from pandas import DataFrame
from copy import deepcopy


def get_unique_systems(dataset, idx_form, idx_target):
    systems = dict()

    for d in dataset:
        if d[idx_form] in systems.keys():
            systems[d[idx_form]]['y'].append(d[idx_target])
        else:
            systems[d[idx_form]] = dict()
            systems[d[idx_form]]['y'] = [d[idx_target]]

    return systems


def generate_dataset(path_dataset, path_new_dataset, idx_form, idx_target, name_target, api_key):
    dataset = pandas.read_excel(path_dataset).values.tolist()
    unique_systems = get_unique_systems(dataset, idx_form, idx_target)
    forms = list(unique_systems.keys())
    new_dataset = dict()

    with MPRester(api_key=api_key) as mpr:
        for i in tqdm(range(0, len(forms))):
            results = mpr.summary.search(formula=forms[i])
            mp_ids = list()

            for r in results:
                mp_id = r.material_id
                struct = r.structure

                cif_writer = CifWriter(struct)
                cif_writer.write_file('{}/{}.cif'.format(path_new_dataset, mp_id))
                mp_ids.append(mp_id)

            new_dataset[forms[i]] = {'possible_structs': mp_ids, name_target: unique_systems[forms[i]]['y']}

    with open('{}/metadata.json'.format(path_new_dataset), 'w') as f:
        json.dump(new_dataset, f)


def to_pretty_forms(path_metadata_file, idx_form):
    dataset = pandas.read_excel(path_metadata_file).values.tolist()

    for d in dataset:
        form_dict = parse_formula(d[idx_form])
        _form_dict = deepcopy(form_dict)

        for e in form_dict.keys():
            if form_dict[e] == 0.667:
                form_dict[e] = 0.666

            _form_dict[e] = int(1000 * form_dict[e])
        gcd = math.gcd(*list(_form_dict.values()))

        for e in _form_dict.keys():
            _form_dict[e] = int(_form_dict[e] / gcd)

        form = ''
        for e in _form_dict.keys():
            if _form_dict[e] == 1:
                form += e
            else:
                form += e + str(_form_dict[e])
        d[idx_form] = form

    DataFrame(dataset).to_excel('dataset.xlsx', index=False, header=False)


def get_k_folds(dataset, k, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    k_folds = list()
    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), k)

    for i in range(0, k):
        idx_train = list(itertools.chain.from_iterable(idx_rand[:i] + idx_rand[i+1:]))
        idx_test = idx_rand[i]
        dataset_train = dataset[idx_train]
        dataset_test = dataset[idx_test]
        k_folds.append([dataset_train, dataset_test])

    return k_folds
