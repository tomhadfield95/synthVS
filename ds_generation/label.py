import argparse
import faulthandler
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from utils import expand_path, save_yaml
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures
from scipy.stats import gamma

FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir, 'BaseFeatures.fdef')))

def vec_to_vec_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def get_atomwise_contributions(mol, contrib_df, ligand_atoms = True):
    
    if ligand_atoms:
        
        atom_indices = []
        for idx, row in contrib_df.iterrows():
            
            found_match = False

            for atom in mol.GetAtoms():

                if vec_to_vec_dist(row['ligand_pos'], np.array(mol.GetConformer().GetAtomPosition(atom.GetIdx()))) < 0.05:
                    atom_indices.append(atom.GetIdx())
                    found_match = True
                    break

            if found_match == False:
                print(Chem.MolToMolBlock(mol))
                print(contrib_df)
                print(idx, row)


        contrib_df['lig_atom_idx'] = atom_indices
        
        atom_contibutions = contrib_df[['lig_atom_idx', 'contribution']].groupby('lig_atom_idx').aggregate(sum)
            
        atom_indices = []
        atom_positions = []

        for idx, atom in enumerate(mol.GetAtoms()):
            atom_indices.append(atom.GetIdx())
            atom_positions.append(np.array(mol.GetConformer().GetAtomPosition(atom.GetIdx())))


        atomwise_df = pd.DataFrame({'lig_atom_idx':atom_indices, 'x':[x[0] for x in atom_positions], 
                              'y':[y[1] for y in atom_positions], 'z': [z[2] for z in atom_positions]})

        contribution = []
        for idx, row in atomwise_df.iterrows():
            c = 0
            for jdx, sow in atom_contibutions.iterrows():
                if row['lig_atom_idx'] == jdx:
                    contribution.append(sow['contribution'])
                    c = 1 #i.e. this atom makes a contribution to the score

            if c == 0:
                contribution.append(0)

        atomwise_df['contribution'] = contribution
        
        return contrib_df, atomwise_df
        
        
    else:
        
        atom_indices = []
        for idx, row in contrib_df.iterrows():

            for atom in mol.GetAtoms():

                if vec_to_vec_dist(row['pharm_pos'], np.array(mol.GetConformer().GetAtomPosition(atom.GetIdx()))) < 0.05:
                    atom_indices.append(atom.GetIdx())
                    
        contrib_df['pharm_atom_idx'] = atom_indices
        
        atom_contibutions = contrib_df[['pharm_atom_idx', 'contribution']].groupby('pharm_atom_idx').aggregate(sum)
            
        atom_indices = []
        atom_positions = []

        for idx, atom in enumerate(mol.GetAtoms()):
            atom_indices.append(atom.GetIdx())
            atom_positions.append(np.array(mol.GetConformer().GetAtomPosition(atom.GetIdx())))


        atomwise_df = pd.DataFrame({'pharm_atom_idx':atom_indices, 'x':[x[0] for x in atom_positions], 
                              'y':[y[1] for y in atom_positions], 'z': [z[2] for z in atom_positions]})

        contribution = []
        for idx, row in atomwise_df.iterrows():
            c = 0
            for jdx, sow in atom_contibutions.iterrows():
                if row['pharm_atom_idx'] == jdx:
                    contribution.append(sow['contribution'])
                    c = 1 #i.e. this atom makes a contribution to the score

            if c == 0:
                contribution.append(0)

        atomwise_df['contribution'] = contribution
        
        return contrib_df, atomwise_df
    





def assign_mol_label(ligand, pharm_mol, threshold=3.5, fname_idx=None, hydrophobic = False, return_contrib_df = False):
    """Assign the labels 0 or 1 to atoms in the pharm/ligand molecules.

    If there is a receptor pharmacophore within the threshold of a matching
    ligand pharmacophore, the class of the atom is 1. If not, it is zero.

    Arguments:
        ligand: RDKit mol object (ligand molecule)
        pharm_mol: RDKit mol object (fake receptor pharmacophores)
        threshold: cutoff for interaction distance which is considered an active
            interaction
        fname_idx: index of ligand and pharm_mol sdf in directory (if supplied)
    """

    

    def get_pharm_indices(mol):
        pharms = ['Hydrophobe', 'Donor', 'Acceptor', 'LumpedHydrophobe']
        pharms_idx_dict = defaultdict(list)
        if mol.GetNumAtoms() < 1:
            return pharms_idx_dict

        mol.AddConformer(mol.GetConformer())
        feats = FACTORY.GetFeaturesForMol(mol)
        for feat in feats:
            if feat.GetFamily() in pharms:
                pharms_idx_dict[feat.GetFamily()] += list(feat.GetAtomIds())

        return pharms_idx_dict

    if pharm_mol is None:
        if fname_idx is not None:
            return fname_idx, []
        return []

    ligand_pharms_indices = get_pharm_indices(ligand)
    ligand_pharms_positions = defaultdict(list)
    # get positions
    for k in ligand_pharms_indices.keys():
        for idx in ligand_pharms_indices[k]:
            ligand_pharms_positions[k].append(
                np.array(ligand.GetConformer().GetAtomPosition(idx)))

    """
    positive_coords = PositionLookup()
    min_distances_to_pharms = []
    """
    
    if not hydrophobic:
    
        positive_coords = []
        for idx, atom in enumerate(pharm_mol.GetAtoms()):
            atom_pos = np.array(
                pharm_mol.GetConformer().GetAtomPosition(idx))
            for atomic_symbol, ligand_pharm_positions in zip(
                    ('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'] +
                                      ligand_pharms_positions['LumpedHydrophobe'],
                                      ligand_pharms_positions['Acceptor'],
                                      ligand_pharms_positions['Donor'])):
                if atom.GetSymbol() == atomic_symbol:
                    for ligand_pharm_position in ligand_pharm_positions:
                        dist = vec_to_vec_dist(ligand_pharm_position, atom_pos)
                        if dist < threshold:
                            positive_coords.append(ligand_pharm_position)
                            positive_coords.append(atom_pos)
        if fname_idx is not None:
            return fname_idx, positive_coords
        return positive_coords
    else:
        
        interaction_score = 0
        
        if return_contrib_df:
            p_positions = []
            l_positions = []
            pairwise_contribution = []
            
            
        
        
        for idx, atom in enumerate(pharm_mol.GetAtoms()):
            atom_pos = np.array(
                pharm_mol.GetConformer().GetAtomPosition(idx))
            for atomic_symbol, ligand_pharm_positions in zip(
                    #('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'] +
                                     # ligand_pharms_positions['LumpedHydrophobe'],
                                     # ligand_pharms_positions['Acceptor'],
                                     # ligand_pharms_positions['Donor'])):
                    ('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'],
                                      ligand_pharms_positions['Acceptor'],
                                      ligand_pharms_positions['Donor'])):
                
                if atom.GetSymbol() == atomic_symbol:
                    for ligand_pharm_position in ligand_pharm_positions:
                        
                        if atomic_symbol == 'C':   
                            
                            ic = interaction_contribution(ligand_pharm_position, atom_pos, hydrophobic = True)
                            interaction_score += ic
                            
                            if return_contrib_df:
                                p_positions.append(atom_pos)
                                l_positions.append(ligand_pharm_position)
                                pairwise_contribution.append(ic)
                            
                        else:
                            ic = interaction_contribution(ligand_pharm_position, atom_pos, hydrophobic = False)
                            interaction_score += ic
                            
                            if return_contrib_df:
                                p_positions.append(atom_pos)
                                l_positions.append(ligand_pharm_position)
                                pairwise_contribution.append(ic)
                                
                        
        if return_contrib_df:
            contrib_df = pd.DataFrame({'pharm_pos':p_positions, 'ligand_pos':l_positions, 'contribution':pairwise_contribution})
            
            
            
            contrib_df, lig_gt = get_atomwise_contributions(ligand, contrib_df, ligand_atoms = True)
            contrib_df, pharm_gt = get_atomwise_contributions(pharm_mol, contrib_df, ligand_atoms = False)

            
            
            return interaction_score, contrib_df, lig_gt, pharm_gt           
        else:
            return interaction_score
    # noinspection PyUnreachableCode
    """
                    if ligand_pharm_position not in positive_coords:
                        positive_coords.append(ligand_pharm_position)
                        min_distances_to_pharms.append(dist)
                    else:
                        min_distances_to_pharms[
                            positive_coords.index(ligand_pharm_position)] 
                            = min(
                            dist, min_distances_to_pharms[
                            positive_coords.index(
                                ligand_pharm_position)])
                    if atom_pos not in positive_coords:
                        positive_coords.append(ligand_pharm_position)
                        min_distances_to_pharms.append(dist)
                    else:
                        min_distances_to_pharms[
                            positive_coords.index(atom_pos)] = min(
                            dist, min_distances_to_pharms[
                            positive_coords.index(
                                atom_pos)])
    if fname_idx is not None:
        return fname_idx, positive_coords, min_distances_to_pharms
    return positive_coords, min_distances_to_pharms    
    """


def interaction_contribution(lig_position, prot_position, hydrophobic = False):
    #For the case where we include hydrophobic pharmacophores, this function assigns
    #a score to each interaction
    #(if the cumulative interaction score is greater than a threshold then we will classify as active, otherwise we will classify as inactive)
    distance = np.linalg.norm(lig_position - prot_position)
    return gamma_score(distance, hydrophobic = hydrophobic)
    #return threshold_score(distance, hydrophobic=hydrophobic)
    
def gamma_score(x, a = 4, hydrophobic = False):
    
    if hydrophobic:
        return 3*gamma.pdf(np.abs(x), a)
    else:
        return 10*gamma.pdf(np.abs(x), a)

def threshold_score(x, hydrophobic = False):
    if hydrophobic:
        if x < 4:
            return 1
        else:
            return 0
    else:
        if x < 4:
            return 10
        else: 
            return 0

def interaction_contribution(lig_position, prot_position, hydrophobic = False):
    #For the case where we include hydrophobic pharmacophores, this function assigns
    #a score to each interaction
    #(if the cumulative interaction score is greater than a threshold then we will classify as active, otherwise we will classify as inactive)
    distance = np.linalg.norm(lig_position - prot_position)
    return gamma_score(distance, hydrophobic = hydrophobic)
    #return threshold_score(distance, hydrophobic=hydrophobic)
    
def gamma_score(x, a = 3, hydrophobic = False):
    
    if hydrophobic:
        return 3*gamma.pdf(np.abs(x), a)
    else:
        return 10*gamma.pdf(np.abs(x), a)

def threshold_score(x, hydrophobic = False):
    if hydrophobic:
        if x < 4:
            return 1
        else:
            return 0
    else:
        if x < 4:
            return 10
        else: 
            return 0


def label_dataset(root, threshold):
    """Use multiprocssing to post-facto label atoms and mols in sdf dataset."""
    faulthandler.enable()
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
        coords_with_positive_label[idx] = [
            [float(i) for i in coords] for coords in positive_coords]
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
    
    
    
    
    
    
    
    
    
    
    
    
'''
    
def assign_mol_label(ligand, pharm_mol, threshold=3.5, fname_idx=None, hydrophobic = False):
    """Assign the labels 0 or 1 to atoms in the pharm/ligand molecules.

    If there is a receptor pharmacophore within the threshold of a matching
    ligand pharmacophore, the class of the atom is 1. If not, it is zero.

    Arguments:
        ligand: RDKit mol object (ligand molecule)
        pharm_mol: RDKit mol object (fake receptor pharmacophores)
        threshold: cutoff for interaction distance which is considered an active
            interaction
        fname_idx: index of ligand and pharm_mol sdf in directory (if supplied)
    """

    def vec_to_vec_dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def get_pharm_indices(mol):
        pharms = ['Hydrophobe', 'Donor', 'Acceptor', 'LumpedHydrophobe']
        pharms_idx_dict = defaultdict(list)
        if mol.GetNumAtoms() < 1:
            return pharms_idx_dict

        mol.AddConformer(mol.GetConformer())
        feats = FACTORY.GetFeaturesForMol(mol)
        for feat in feats:
            if feat.GetFamily() in pharms:
                pharms_idx_dict[feat.GetFamily()] += list(feat.GetAtomIds())

        return pharms_idx_dict

    if pharm_mol is None:
        if fname_idx is not None:
            return fname_idx, []
        return []

    ligand_pharms_indices = get_pharm_indices(ligand)
    ligand_pharms_positions = defaultdict(list)
    # get positions
    for k in ligand_pharms_indices.keys():
        for idx in ligand_pharms_indices[k]:
            ligand_pharms_positions[k].append(
                np.array(ligand.GetConformer().GetAtomPosition(idx)))

    """
    positive_coords = PositionLookup()
    min_distances_to_pharms = []
    """
    
    if not hydrophobic:
    
        positive_coords = []
        for idx, atom in enumerate(pharm_mol.GetAtoms()):
            atom_pos = np.array(
                pharm_mol.GetConformer().GetAtomPosition(idx))
            for atomic_symbol, ligand_pharm_positions in zip(
                    ('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'] +
                                      ligand_pharms_positions['LumpedHydrophobe'],
                                      ligand_pharms_positions['Acceptor'],
                                      ligand_pharms_positions['Donor'])):
                if atom.GetSymbol() == atomic_symbol:
                    for ligand_pharm_position in ligand_pharm_positions:
                        dist = vec_to_vec_dist(ligand_pharm_position, atom_pos)
                        if dist < threshold:
                            positive_coords.append(ligand_pharm_position)
                            positive_coords.append(atom_pos)
        if fname_idx is not None:
            return fname_idx, positive_coords
        return positive_coords
    else:
        
        interaction_score = 0
        
        for idx, atom in enumerate(pharm_mol.GetAtoms()):
            atom_pos = np.array(
                pharm_mol.GetConformer().GetAtomPosition(idx))
            for atomic_symbol, ligand_pharm_positions in zip(
                    #('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'] +
                                     # ligand_pharms_positions['LumpedHydrophobe'],
                                     # ligand_pharms_positions['Acceptor'],
                                     # ligand_pharms_positions['Donor'])):
                    ('C', 'O', 'N'), (ligand_pharms_positions['Hydrophobe'],
                                      ligand_pharms_positions['Acceptor'],
                                      ligand_pharms_positions['Donor'])):
                
                if atom.GetSymbol() == atomic_symbol:
                    for ligand_pharm_position in ligand_pharm_positions:
                        
                        if atomic_symbol == 'C':    
                            interaction_score += interaction_contribution(ligand_pharm_position, atom_pos, hydrophobic = True)
                        else:
                            interaction_score += interaction_contribution(ligand_pharm_position, atom_pos, hydrophobic = False)
                        
                        
        return interaction_score
    
    
'''
    
    
    
    
