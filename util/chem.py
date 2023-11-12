import numpy
import torch
import json
from mendeleev.fetch import fetch_table
from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances
from chemparse import parse_formula
from torch_geometric.data import Data
from rdkit import Chem


atom_nums = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
}
atom_syms = {v: k for k, v in atom_nums.items()}
elem_attr_names = [
    'atomic_number', 'atomic_weight', 'atomic_radius', 'atomic_volume', 'atomic_weight',
    'covalent_radius_pyykko', 'density', 'dipole_polarizability', 'electron_affinity', 'en_allen',
    'en_ghosh', 'en_pauling', 'evaporation_heat', 'fusion_heat', 'heat_of_formation', 'mendeleev_number',
    'period', 'thermal_conductivity', 'vdw_radius'
]
first_ion_energies = [
    1312, 2372.3, 520.2, 899.5, 800.6, 1086.5, 1402.3, 1313.9, 1681, 2080.7,
    495.8, 737.7, 577.5, 786.5, 1011.8, 999.6, 1251.2, 1520.6, 418.8, 589.8,
    633.1, 658.8, 650.9, 652.9, 717.3, 762.5, 760.4, 737.1, 745.5, 906.4,
    578.8, 762, 947, 941, 1139.9, 1350.8, 403, 549.5, 600, 640.1,
    652.1, 684.3, 702, 710.2, 719.7, 804.4, 731, 867.8, 558.3, 708.6,
    834, 869.3, 1008.4, 1170.4, 375.7, 502.9, 538.1, 534.4, 527, 533.1,
    540, 544.5, 547.1, 593.4, 565.8, 573, 581, 589.3, 596.7, 603.4,
    523.5, 658.5, 761, 770, 760, 840, 880, 870, 890.1, 1007.1,
    589.4, 715.6, 703, 812.1, 899.003, 1037, 380, 509.3, 499, 587,
    568, 597.6, 604.5, 584.7, 578, 581, 601, 608, 619, 627,
    635, 642, 470, 580, 665, 757, 740, 730, 800, 960,
    568, 584, 597, 585, 578, 581, 601, 608, 619, 627,
    635, 642, 478.57, 490, 640, 730, 660, 750, 840
]
cat_bond_types = [
    'UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE',
    'QUINTUPLE', 'HEXTUPLE', 'ONEANDAHALF', 'TWOANDAHALF', 'THREEANDAHALF',
    'FOURANDAHALF', 'FIVEANDAHALF', 'AROMATIC', 'IONIC', 'HYDROGEN',
    'THREECENTER', 'DATIVEONE', 'DATIVE', 'DATIVEL', 'DATIVER', 'OTHER', 'ZERO'
]


def get_one_hot_feat(hot_category, categories):
    one_hot_feat = dict()
    for cat in categories:
        one_hot_feat[cat] = 0

    if hot_category in categories:
        one_hot_feat[hot_category] = 1

    return numpy.array(list(one_hot_feat.values()))


def load_elem_attrs(path_elem_attr=None):
    if path_elem_attr is None:
        elem_attrs = numpy.nan_to_num(numpy.array(fetch_table('elements')[elem_attr_names]))
        ion_energies = numpy.array(first_ion_energies[:elem_attrs.shape[0]]).reshape(-1, 1)
        elem_attrs = scale(numpy.hstack([elem_attrs, ion_energies]))

        return torch.tensor(elem_attrs[:len(atom_nums), :], dtype=torch.float)
    else:
        with open(path_elem_attr) as json_file:
            elem_attr = json.load(json_file)
            elem_attrs = [torch.tensor(elem_attr[elem], dtype=torch.float) for elem in atom_nums.keys()]

        return torch.vstack(elem_attrs)[:len(atom_nums), :]


def get_form_vec(form, elem_attrs):
    elem_dict = parse_formula(form)
    wt_sum_feats = torch.zeros(elem_attrs.shape[1])
    list_atom_feats = list()
    sum_elem_nums = sum([float(elem_dict[e]) for e in elem_dict.keys()])

    for e in elem_dict.keys():
        atom_feats = elem_attrs[atom_nums[e] - 1, :]
        list_atom_feats.append(atom_feats)
        wt_sum_feats += (float(elem_dict[e]) / sum_elem_nums) * atom_feats
    form_atom_feats = torch.vstack(list_atom_feats)

    form_vec = torch.hstack([wt_sum_feats,
                             torch.min(form_atom_feats, dim=0)[0],
                             torch.max(form_atom_feats, dim=0)[0]])

    return form_vec


def get_mol_graph(mol, elem_attrs, y, add_h=False):
    try:
        atom_feats = list()
        bonds = list()
        bond_feats = list()

        if add_h:
            mol = Chem.AddHs(mol)

        if mol is None:
            return None

        for atom in mol.GetAtoms():
            elem_attr = elem_attrs[atom.GetAtomicNum() - 1, :]
            mem_aromatic = 1 if atom.GetIsAromatic() else 0
            degree = atom.GetDegree()
            n_hs = atom.GetTotalNumHs()
            atom_feats.append(numpy.hstack([elem_attr, mem_aromatic, degree, n_hs]))

        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_feats.append(get_one_hot_feat(str(bond.GetBondType()), cat_bond_types))
            bonds.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            bond_feats.append(get_one_hot_feat(str(bond.GetBondType()), cat_bond_types))

        if len(bonds) == 0:
            return None

        atom_feats = torch.tensor(numpy.vstack(atom_feats), dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        bond_feats = torch.tensor(numpy.vstack(bond_feats), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1, 1)

        return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, y=y)
    except RuntimeError:
        return None


def get_crystal_graph(struct, elem_attrs, rbf_means, fvec, form_eng, target, atomic_cutoff=4.0):
    try:
        atom_coord, atom_feats, ans = get_atom_info(struct, elem_attrs, atomic_cutoff)
        bonds, bond_feats = get_bond_info(atom_coord, rbf_means, atomic_cutoff)

        if bonds is None:
            return None

        crystal_graph = Data(x=torch.tensor(atom_feats, dtype=torch.float),
                             edge_index=torch.tensor(bonds, dtype=torch.long).t().contiguous(),
                             edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                             fvec=fvec,
                             form_eng=torch.tensor(form_eng, dtype=torch.float).view(1, 1),
                             y=torch.tensor(target, dtype=torch.float).view(1, 1))

        return crystal_graph
    except AssertionError:
        return None


def get_atom_info(crystal, elem_attrs, atomic_cutoff):
    atoms = list(crystal.atomic_numbers)
    atom_coord = list()
    atom_feats = list()
    list_nbrs = crystal.get_all_neighbors(atomic_cutoff)

    coords = dict()
    for coord in list(crystal.cart_coords):
        coord_key = ','.join(list(coord.astype(str)))
        coords[coord_key] = True

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))
            if coord_key not in coords.keys():
                coords[coord_key] = True
                atoms.append(atom_nums[nbrs[j][0].species_string])

    for coord in coords.keys():
        atom_coord.append(numpy.array([float(x) for x in coord.split(',')]))
    atom_coord = numpy.vstack(atom_coord)

    for i in range(0, len(atoms)):
        atom_feats.append(elem_attrs[atoms[i]-1, :])
    atom_feats = numpy.vstack(atom_feats).astype(float)

    return atom_coord, atom_feats, atoms


def get_bond_info(atom_coord, rbf_means, atomic_cutoff):
    bonds = list()
    bond_feats = list()
    pdist = pairwise_distances(atom_coord)

    for i in range(0, atom_coord.shape[0]):
        for j in range(0, atom_coord.shape[0]):
            if i != j and pdist[i, j] < atomic_cutoff:
                bonds.append([i, j])
                bond_feats.append(rbf(numpy.full(rbf_means.shape[0], pdist[i, j]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None
    else:
        bonds = numpy.vstack(bonds)
        bond_feats = numpy.vstack(bond_feats)

        return bonds, bond_feats


def rbf(data, mu, beta):
    return numpy.exp(-(data - mu)**2 / beta**2)
