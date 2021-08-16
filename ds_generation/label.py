import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from point_vs.utils import expand_path, save_yaml
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures

FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir, 'BaseFeatures.fdef')))


def assign_mol_label(ligand, pharm_mol, threshold=3.5, idx=None):
    """If there is a pharmacophore within the threshold of a matching ligand
    pharmacophore, then we return a label of 1 else 0"""

    def vec_to_vec_dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def get_pharm_indices(mol):
        pharms = ['Hydrophobe', 'Donor', 'Acceptor', 'LumpedHydrophobe']
        pharms_idx_dict = defaultdict(list)

        feats = FACTORY.GetFeaturesForMol(mol)
        for feat in feats:
            if feat.GetFamily() in pharms:
                pharms_idx_dict[feat.GetFamily()] += list(feat.GetAtomIds())

        return pharms_idx_dict

    if pharm_mol is None:
        print('pharm_mol is None', idx)
        if idx is not None:
            return idx, []
        return []

    ligand_pharms_indices = get_pharm_indices(ligand)
    ligand_pharms_positions = defaultdict(list)
    # get positions
    for k in ligand_pharms_indices.keys():
        for idx in ligand_pharms_indices[k]:
            ligand_pharms_positions[k].append(
                np.array(ligand.GetConformer().GetAtomPosition(idx)))

    positive_coords = []
    for atom in pharm_mol.GetAtoms():
        atom_pos = np.array(
            pharm_mol.GetConformer().GetAtomPosition(atom.GetIdx()))
        if atom.GetSymbol() == 'C':
            raise
        for atomic_symbol, ligand_pharm_positions in zip(('C', 'O', 'N'), zip(
                ligand_pharms_positions['Hydrophobe'] +
                ligand_pharms_positions['LumpedHydrophobe'],
                ligand_pharms_positions['Acceptor'],
                ligand_pharms_positions['Donor'])):
            if atom.GetSymbol() == atomic_symbol:
                if atomic_symbol not in ('O', 'N'):
                    raise
                for ligand_pharm_position in ligand_pharm_positions:
                    if vec_to_vec_dist(
                            ligand_pharm_position, atom_pos) < threshold:
                        positive_coords.append(ligand_pharm_position)
                        positive_coords.append(atom_pos)

    print(idx)
    if idx is not None:
        return idx, positive_coords
    return positive_coords


def label_dataset(root, threshold):
    root = expand_path(root)
    mol_labels = {}
    coords_with_positive_label = {}
    indices, pharm_mols, lig_mols = [], [], []
    for lig_sdf in Path(root, 'ligands').glob('*.sdf'):
        idx = int(Path(lig_sdf.name).stem.split('lig')[-1])
        pharm_sdf = str(root / 'pharmacophores' / 'pharm{}.sdf'.format(idx))
        lig_mols.append(Chem.SDMolSupplier(str(lig_sdf))[0])
        pharm_mols.append(Chem.SDMolSupplier(str(pharm_sdf))[0])
        indices.append(idx)
    print('SDFs loaded.')
    thresholds = [threshold] * len(indices)
    results = Pool().map(
        assign_mol_label, lig_mols, pharm_mols, thresholds, indices)
    print('SDFs processed.')
    for res in results:
        idx = res[0]
        positive_coords = res[1]
        coords_with_positive_label[idx] = positive_coords
        mol_labels[idx] = int(len(positive_coords) > 0)
    print('Results constructed.')
    return coords_with_positive_label, mol_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str,
                        help='Root location of ligand and pharmacophore sdf '
                             'files')
    parser.add_argument('threshold', type=float,
                        help='Angstrom cutoff for positive labelling')
    args = parser.parse_args()

    root = expand_path(args.root)
    atomic_labels, mol_labels = label_dataset(root, args.threshold)
    save_yaml(atomic_labels, root / 'atomic_labels.yaml')
    save_yaml(mol_labels, root / 'labels.yaml')
