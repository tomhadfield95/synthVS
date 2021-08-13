import numpy as np
from rdkit import Chem


def define_box_around_ligand(positions, slack=5):

    x_min, x_max = min(positions[:, 0]) - slack, max(positions[:, 0]) + slack
    y_min, y_max = min(positions[:, 1]) - slack, max(positions[:, 1]) + slack
    z_min, z_max = min(positions[:, 2]) - slack, max(positions[:, 2]) + slack

    return {'X': [x_min, x_max], 'Y': [y_min, y_max], 'Z': [z_min, z_max]}


def sample_pharmacophores_box(positions, pharm_dict=None):
    if pharm_dict is None:
        pharm_dict = {0: 'Acceptor', 1: 'Donor', 2: 'Hydrophobe'}
    # Increasing diagonal elements on cov matrix increases dispersion from
    # ligand atom

    box_params = define_box_around_ligand(positions)

    # sample random pharmacophore type
    # FOR NOW WE WON'T THINK ABOUT HYDROPHOBIC INTERACTIONS AND JUST CONSIDER
    # HYDROGEN BONDS
    pharm_type = np.random.choice([0, 1, 2], p=[0.5, 0.5, 0.0])

    # sample pharmacophore location

    x_pos = np.random.uniform(low=box_params['X'][0], high=box_params['X'][1])
    y_pos = np.random.uniform(low=box_params['Y'][0], high=box_params['Y'][1])
    z_pos = np.random.uniform(low=box_params['Z'][0], high=box_params['Z'][1])

    random_pharmacophore_pos = np.array([x_pos, y_pos, z_pos])

    return random_pharmacophore_pos, pharm_dict[pharm_type]


def create_pharmacophore_mol_by_area(mol, area_coef=0.25):
    # Want the number of pharmacophores to be proportional to the area of the
    # box around the ligand

    # Get Atom Positions
    conf = mol.GetConformer()
    positions = np.array(
        [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    box_params = define_box_around_ligand(positions)
    box_area = (box_params['X'][1] - box_params['X'][0]) *\
               (box_params['Y'][1] - box_params['Y'][0]) * (
            box_params['Z'][1] - box_params['Z'][0])

    num_pharmacophores = int(np.ceil(box_area * area_coef))

    # Create empty molecule
    pharm_mol = Chem.RWMol()
    positions = []

    # Get Atom Positions
    conf = mol.GetConformer()
    orig_positions = np.array(
        [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    for i in range(num_pharmacophores):
        pos, pharm_type = sample_pharmacophores_box(orig_positions)
        pharm_mol = add_pharm_atom(pharm_mol, pharm_type)
        positions.append(pos)

    conf = Chem.Conformer(num_pharmacophores)

    for i in range(num_pharmacophores):
        conf.SetAtomPosition(i, positions[i])

    pharm_mol.AddConformer(conf)

    return pharm_mol


def add_pharm_atom(pharm_mol, pharm_type):
    if pharm_type == 'Acceptor':
        pharm_mol.AddAtom(Chem.Atom(8))  # Oxygen atom denotes HBA
    elif pharm_type == 'Donor':
        pharm_mol.AddAtom(Chem.Atom(7))  # Nitrogen atom denotes HBD
    elif pharm_type == 'Hydrophobe':
        pharm_mol.AddAtom(Chem.Atom(6))  # Carbon atom denotes Hyrdophobic
    return pharm_mol


# create pharmacophores
def sample_atom(mol):
    num_atoms = mol.GetNumHeavyAtoms()
    idx = np.random.randint(0, high=num_atoms, dtype=int)
    return idx


def sample_pharmacophores(mol, pharm_dict=None, cov=np.identity(3) * 3):
    if pharm_dict is None:
        pharm_dict = {0: 'Acceptor', 1: 'Donor', 2: 'Hydrophobe'}
    # Increasing diagonal elements on cov matrix increases dispersion from
    # ligand atom

    # sample random atom
    if pharm_dict is None:
        pharm_dict = {0: 'Acceptor', 1: 'Donor', 2: 'Hydrophobe'}
    random_idx = sample_atom(mol)

    # sample random pharmacophore type
    # FOR NOW WE WON'T THINK ABOUT HYDROPHOBIC INTERACTIONS AND JUST CONSIDER
    # HYDROGEN BONDS
    pharm_type = np.random.choice([0, 1, 2], p=[0.5, 0.5, 0.0])

    # sample pharmacophore location
    random_atom_pos = np.array(mol.GetConformer().GetAtomPosition(random_idx))
    random_pharmacophore_pos = np.random.multivariate_normal(random_atom_pos, cov)

    return random_pharmacophore_pos, random_idx, pharm_dict[pharm_type]


def create_pharmacophore_mol_by_absolute(mol, num_pharmacophores=10):
    # Create empty molecule
    pharm_mol = Chem.RWMol()
    positions = []

    for i in range(num_pharmacophores):
        pos, idx, pharm_type = sample_pharmacophores(mol)

        pharm_mol = add_pharm_atom(pharm_mol, pharm_type)
        positions.append(pos)

    conf = Chem.Conformer(num_pharmacophores)

    for i in range(num_pharmacophores):
        conf.SetAtomPosition(i, positions[i])

    pharm_mol.AddConformer(conf)

    return pharm_mol


def create_pharmacophore_mol(mol, num_pharmacophores, area_coef):
    assert (num_pharmacophores is None) + (area_coef is None) == 1, (
        'num_pharmacophores and area_coeff are mutually exclusive arguments')
    if num_pharmacophores is None:
        return create_pharmacophore_mol_by_area(mol, area_coef)
    return create_pharmacophore_mol_by_absolute(mol, num_pharmacophores)
