import numpy as np
from rdkit import Chem


# create pharmacophores
def sample_atom(mol):
    num_atoms = mol.GetNumHeavyAtoms()
    idx = np.random.randint(0, high=num_atoms, dtype=int)
    return idx


def sample_pharmacophores(mol, pharm_dict=None, cov=np.identity(3) * 3):
    if pharm_dict is None:
        pharm_dict = {0: 'Acceptor', 1: 'Donor', 2: 'Hydrophobe'}
    # Increasing diagonal elements on cov matrix increases dispersion from ligand atom

    # sample random atom
    if pharm_dict is None:
        pharm_dict = {0: 'Acceptor', 1: 'Donor', 2: 'Hydrophobe'}
    random_idx = sample_atom(mol)

    # sample random pharmacophore type
    pharm_type = np.random.choice([0, 1, 2], p=[0.5, 0.5, 0.0])
    # FOR NOW WE WON'T THINK ABOUT HYDROPHOBIC INTERACTIONS AND JUST CONSIDER HYDROGEN BONDS

    # sample pharmacophore location
    random_atom_pos = np.array(mol.GetConformer().GetAtomPosition(random_idx))
    random_pharmacophore_pos = np.random.multivariate_normal(random_atom_pos, cov)

    return random_pharmacophore_pos, random_idx, pharm_dict[pharm_type]


def create_pharmacophore_mol(mol, num_pharmacophores=10):
    # Create empty molecule
    pharm_mol = Chem.RWMol()
    positions = []

    for i in range(num_pharmacophores):

        pos, idx, pharm_type = sample_pharmacophores(mol)

        if pharm_type == 'Acceptor':
            pharm_mol.AddAtom(Chem.Atom(8))  # Oxygen atom denotes HBA
        if pharm_type == 'Donor':
            pharm_mol.AddAtom(Chem.Atom(7))  # Nitrogen atom denotes HBD
        if pharm_type == 'Hydrophobe':
            pharm_mol.AddAtom(Chem.Atom(6))  # Carbon atom denotes Hyrdophobic

        positions.append(pos)

    conf = Chem.Conformer(num_pharmacophores)

    for i in range(num_pharmacophores):
        conf.SetAtomPosition(i, positions[i])

    pharm_mol.AddConformer(conf)

    return pharm_mol
