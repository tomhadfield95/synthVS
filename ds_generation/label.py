from pathlib import Path

import numpy as np
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


def assign_label(ligand, pharm_mol, threshold=3.5):
    # If there is a pharmacophore within the threshold of a matching ligand pharmacophore, then we return a label of 1
    # Otherwise a label of 0

    def vec_to_vec_dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def get_pharm_indices(mol):
        pharms = ['Hydrophobe', 'Donor', 'Acceptor', 'LumpedHydrophobe']
        pharms_idx_dict = {'LumpedHydrophobe': [], 'Hydrophobe': [], 'Acceptor': [], 'Donor': []}

        fdef_name = str(Path(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        feats = factory.GetFeaturesForMol(mol)
        for feat in feats:
            if feat.GetFamily() in pharms:
                pharms_idx_dict[feat.GetFamily()] = pharms_idx_dict[feat.GetFamily()] + list(feat.GetAtomIds())

        return pharms_idx_dict

    ligand_pharms_indices = get_pharm_indices(ligand)
    ligand_pharms_positions = {}
    # get positions
    for k in ligand_pharms_indices.keys():
        ligand_pharms_positions[k] = []

        for idx in ligand_pharms_indices[k]:
            ligand_pharms_positions[k].append(np.array(ligand.GetConformer().GetAtomPosition(idx)))

    for atom in pharm_mol.GetAtoms():
        atom_pos = np.array(pharm_mol.GetConformer().GetAtomPosition(atom.GetIdx()))

        # Check Hydrophobic
        if atom.GetSymbol() == 'C':
            distances = [vec_to_vec_dist(atom_pos, x) for x in ligand_pharms_positions['Hydrophobe']]
            distances = distances + [vec_to_vec_dist(atom_pos, x) for x in ligand_pharms_positions['LumpedHydrophobe']]
            if len(distances) and min(distances) < threshold:
                return ligand, pharm_mol, 1

                # Check Acceptor
        if atom.GetSymbol() == 'O':
            distances = [vec_to_vec_dist(atom_pos, x) for x in ligand_pharms_positions['Acceptor']]
            if len(distances) and min(distances) < threshold:
                # return ligand, pharmMol, 2
                return ligand, pharm_mol, 1

        # Check Donor
        if atom.GetSymbol() == 'N':
            distances = [vec_to_vec_dist(atom_pos, x) for x in ligand_pharms_positions['Donor']]
            if len(distances) and min(distances) < threshold:
                # return ligand, pharmMol, 3
                return ligand, pharm_mol, 1

    return ligand, pharm_mol, 0  # To get to this stage, must be no pharmacophores in close proximity.
