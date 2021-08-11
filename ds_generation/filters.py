import numpy as np
from rdkit import Chem


# Post-pharmacophore generation filters
def vec_to_vec_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def pharm_ligand_distance_filter(ligand, pharm_mol, threshold=2):
    # For each atom in the pharmMol, we want to compute the distance to each ligand atom.
    # If a pharm atom is closer to a ligand atom than the threshold, then we remove it

    if pharm_mol is None or pharm_mol.GetNumHeavyAtoms() < 1:
        return None

    pharm_mol_conf = pharm_mol.GetConformer()
    ligand_conf = ligand.GetConformer()

    pharm_mol_positions = [np.array(pharm_mol_conf.GetAtomPosition(i)) for
                           i in range(pharm_mol.GetNumHeavyAtoms())]
    lig_positions = [np.array(ligand_conf.GetAtomPosition(i)) for
                     i in range(ligand.GetNumHeavyAtoms())]

    new_pharm_mol = Chem.RWMol()
    pharm_mol_positions_to_keep = []

    for i in range(pharm_mol.GetNumHeavyAtoms()):

        # Compare pharmacophore position to each ligand atom
        distances = [vec_to_vec_dist(pharm_mol_positions[i], lp) for
                     lp in lig_positions]

        if min(distances) > threshold:
            # Pharmacophore isn't too close to any ligand atoms.
            atom_type = pharm_mol.GetAtomWithIdx(i).GetAtomicNum()
            new_pharm_mol.AddAtom(Chem.Atom(atom_type))
            pharm_mol_positions_to_keep.append(pharm_mol_positions[i])

    # Now we want to add the coordinates to the mol
    conf = Chem.Conformer(new_pharm_mol.GetNumHeavyAtoms())

    for idx, pos in enumerate(pharm_mol_positions_to_keep):
        conf.SetAtomPosition(idx, pos)

    new_pharm_mol.AddConformer(conf)

    return new_pharm_mol


def pharm_pharm_distance_filter(pharm_mol, threshold=3):
    # For each atom in the pharmMol, we want to ensure we're not too close to any of the other atoms
    # Define a list with one atom in it.
    # For another atom, if it's further away from the atom in the list than the threshold, then add it to the list.
    # Otherwise discard it.

    pharm_mol_conf = pharm_mol.GetConformer()
    pharm_mol_positions = [np.array(pharm_mol_conf.GetAtomPosition(i)) for
                           i in range(pharm_mol.GetNumHeavyAtoms())]

    new_pharm_mol = Chem.RWMol()
    if pharm_mol is None or pharm_mol.GetNumHeavyAtoms() < 1:
        return None
    pharm_mol_positions_to_keep = []

    # Add the first atom
    new_pharm_mol.AddAtom(Chem.Atom(pharm_mol.GetAtomWithIdx(0).GetAtomicNum()))
    pharm_mol_positions_to_keep.append(pharm_mol_positions[0])

    for i in range(1, pharm_mol.GetNumHeavyAtoms()):

        distances = [vec_to_vec_dist(pharm_mol_positions[i], pp) for
                     pp in pharm_mol_positions_to_keep]

        if min(distances) > threshold:
            new_pharm_mol.AddAtom(
                Chem.Atom(pharm_mol.GetAtomWithIdx(i).GetAtomicNum()))
            pharm_mol_positions_to_keep.append(pharm_mol_positions[i])

    # Now we want to add the coordinates to the mol
    conf = Chem.Conformer(new_pharm_mol.GetNumHeavyAtoms())

    for idx, pos in enumerate(pharm_mol_positions_to_keep):
        conf.SetAtomPosition(idx, pos)

    new_pharm_mol.AddConformer(conf)

    return new_pharm_mol


def sample_from_pharmacophores(pharm_mol, poisson_mean=10):
    # Sample from a poisson distribution to get the number of pharmacophores to include in the final mol
    # Then sample that many mols

    if pharm_mol is None or pharm_mol.GetNumHeavyAtoms() == 0:
        return None

    pharm_mol_conf = pharm_mol.GetConformer()
    pharm_mol_positions = [np.array(pharm_mol_conf.GetAtomPosition(i)) for i in range(pharm_mol.GetNumHeavyAtoms())]

    num_to_sample = int(np.random.poisson(lam=poisson_mean))

    if num_to_sample > pharm_mol.GetNumHeavyAtoms():
        return pharm_mol  # return all the atoms

    sample_idx = np.random.choice(list(range(pharm_mol.GetNumHeavyAtoms())), num_to_sample, False)  # Atom indices

    new_pharm_mol = Chem.RWMol()
    pharm_mol_positions_to_keep = []

    for idx in sample_idx:
        new_pharm_mol.AddAtom(Chem.Atom(pharm_mol.GetAtomWithIdx(int(idx)).GetAtomicNum()))
        pharm_mol_positions_to_keep.append(pharm_mol_positions[idx])

    # Now we want to add the coordinates to the mol
    conf = Chem.Conformer(new_pharm_mol.GetNumHeavyAtoms())

    for idx, pos in enumerate(pharm_mol_positions_to_keep):
        conf.SetAtomPosition(idx, pos)

    new_pharm_mol.AddConformer(conf)

    return new_pharm_mol
