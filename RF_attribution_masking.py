from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score
from rdkit import RDConfig, Chem
from rdkit.Chem import ChemicalFeatures
from collections import defaultdict
from pathlib import Path
import argparse

from ds_generation.label import assign_mol_label


import multiprocessing as mp
from multiprocessing import Pool

FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir, 'BaseFeatures.fdef')))


import pickle
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import tempfile
import shutil


import oddt
from oddt import fingerprints


def checkArray(single, listOfArrays):

    for idx, arr in enumerate(listOfArrays):

        if single[0] == arr[0] and single[1] == arr[1] and single[2] == arr[2]:
            return True

    return False


def get_features_of_interest(atom_of_interest_idx, lig, bits_info):
    
    #for a specific atom in a ligand, find which of the PLEC bits are associated with that atom
    
    #atom_of_interest_idx - the index of the atom in the ligand
    #lig the (RDKit) ligand
    #bits_info - the bits_info associated with the PLCE fingerprint
    
    #Return - a list containing the indices of the PLEC bits associated with the atom.
    
    features_of_interest = []

    for feature_id in bits_info.keys():

        bit_info_record = list(bits_info[feature_id])
        for b in bit_info_record:
            ligand_root_atom_idx = b[0]
            ligand_depth = b[1]



            #print(ligand_root_atom_idx_rdkit,atom_of_interest_idx_rdkit)


            if ligand_root_atom_idx == atom_of_interest_idx:
                features_of_interest.append(feature_id)
            else:

                path_distance = Chem.rdmolops.GetShortestPath(lig, ligand_root_atom_idx, atom_of_interest_idx)

                if len(path_distance) <= ligand_depth + 1: #Path distance includes both terminal atoms but we only want to include one of them.
                    features_of_interest.append(feature_id)

            
    return list(set(features_of_interest))


def get_trees_using_features(features, random_forest, feature_list):
    
    #Find which trees in a random forest use a specified feature for prediction.
    
    #feature - the name of the feature in the df provided to the random forest model
    #random forest - a fitted RandomForestClassifier
    #feature_list - the list of all feature names used to fit the model.
    
    
    trees_using_features = []
    
    for feature in features:
        #feature_idx = feature_list.index(feature)
        feature_idx = feature #should work in cases where we're working with numpy arrays not dataframes

        for idx, tree in enumerate(random_forest.estimators_):
            #We determine whether a feature was used in a tree if its feature importance was non-zero
            feature_importance = tree.feature_importances_[feature_idx]
            
            #if idx == 0:
                #print(tree.feature_importances_)

            if feature_importance > 0.01:
                trees_using_features.append(idx)
            
    return list(set(trees_using_features))
    

    
def attribution_score_for_features_and_example(x, features, random_forest, feature_list):

    #x - the data
    #feature - the name of the feature in the df provided to the random forest model
    #random forest - a fitted RandomForestClassifier
    #feature_list - the list of all feature names used to fit the model.
    
    
    #Get list of trees which use the feature we're interested in
    trees_using_features = get_trees_using_features(features, random_forest, feature_list)
    #print('Trees using features')
    #print(trees_using_features)

    
    #Obtain attribution score
    
    predictions_using_features = []
    predictions_not_using_features = []

    for idx, tree in enumerate(random_forest.estimators_):

        if idx in trees_using_features:
            predictions_using_features.append(tree.predict(x.reshape(1,-1))[0])
        else:
            predictions_not_using_features.append(tree.predict(x.reshape(1,-1))[0])

    if len(predictions_using_features) == 0 or len(predictions_not_using_features) == 0:
        return 0

    mean_using_features = np.mean(predictions_using_features)
    mean_not_using_features = np.mean(predictions_not_using_features)
    
    return mean_using_features - mean_not_using_features

def rdmol_to_dataframe(mol):
    if mol is None or mol.GetNumHeavyAtoms() < 1:
        return pd.DataFrame({
            'x': [],
            'y': [],
            'z': [],
            'type': []
        })

    conf = mol.GetConformer()
    positions = np.array([np.array(conf.GetAtomPosition(i)) for
                          i in range(mol.GetNumHeavyAtoms())])
    atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for
                  i in range(mol.GetNumHeavyAtoms())]

    if len(atom_types) == 1:
        positions = positions.reshape((1, 3))

    df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'type': atom_types
    })
    if isinstance(mol, Chem.RWMol):
        df['type'] = df['type'].map({8: 0, 7: 1, 6: 2})
    return df

def vector_distance(x, y):
    diff = np.subtract(x,y)
    return np.linalg.norm(diff)


def create_gt_df(ligand, pharm, threshold = 3.5, hydrophobic = False):
    

    if not hydrophobic:
        positive_coords = assign_mol_label(ligand, pharm, threshold = threshold)
    
        lig_df = rdmol_to_dataframe(ligand)
        pharm_df = rdmol_to_dataframe(pharm)
    
    
        lig_gt = [0]*lig_df.shape[0]
        pharm_gt = [0]*pharm_df.shape[0]
    
    
        for idx, row in lig_df.iterrows():
        
            for jdx, coord in enumerate(positive_coords):
            
                if vector_distance(np.array([row['x'], row['y'], row['z']]), coord) < 0.05:
                    lig_gt[idx] = 1
                
        for idx, row in pharm_df.iterrows():
        
            for jdx, coord in enumerate(positive_coords):
            
                if vector_distance(np.array([row['x'], row['y'], row['z']]), coord) < 0.05:
                    pharm_gt[idx] = 1
                
        lig_df['binding'] = lig_gt
        pharm_df['binding'] = pharm_gt
    
        return lig_df, pharm_df

    else:
        interaction_score, contrib_df, lig_df, pharm_df = assign_mol_label(ligand, pharm, hydrophobic = True, return_contrib_df = True)

        return lig_df, pharm_df

'''
    
def assign_mol_label(ligand, pharm_mol, threshold=3.5, fname_idx=None):
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
   

'''


#Get Labels
#labels = pd.read_csv('/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/labels.yaml', header = None)
#y = [int(l.split(': ')[1]) for l in labels[0]]

#indices = np.arange(len(y))

##Get features
#X = np.load('zinc250k_PLEC.npy')
#X_train, X_test, y_train, y_test, indices_train, indices_test = tts(X, y, indices, test_size=0.2, random_state=42)
#load saved model

#with open('savedRFModel.pickle', 'rb') as f:
#    clf = pickle.load(f)


def swap_for_dummy_atom(mol, idx):
    #Input a molecule and atom index and return a molecule where the atom with index idx has been replaced by a dummy atom

    rwmol = Chem.RWMol(mol)
    rwmol.GetAtomWithIdx(idx).SetAtomicNum(0)
    
    return rwmol

def rf_pred(lig_path, prot_path, RF_model, distance_cutoff = 4):

    #Get PLEC fingerprint
    lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lig_path)][0]
    prot_oddt = [l for l in oddt.toolkit.readfile('sdf', prot_path)][0]

    fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein=0, sparse = False, size = 1024, distance_cutoff = distance_cutoff)

    pred = RF_model.predict_proba(fp_dense.reshape(1,-1))[0][1]

    return pred


def get_masking_score(lig_path, prot_path, idx, lig_masked_path, RF_model, plec_threshold):

    lig = Chem.MolFromMolFile(lig_path)
    all_atom_score = rf_pred(lig_path, prot_path, RF_model)
    lig_masked = swap_for_dummy_atom(lig, idx)
    Chem.MolToMolFile(lig_masked, lig_masked_path)
    masked_score = rf_pred(lig_masked_path, prot_path, RF_model = RF_model, distance_cutoff = plec_threshold)
        
    #print(all_atom_score, masked_score)

    return all_atom_score - masked_score


'''
def get_highest_ranking_positive_masking(positive_idx):

    lig_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/ligands/lig{indices_test[positive_idx]}.sdf'
    prot_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/pharmacophores/pharm{indices_test[positive_idx]}.sdf'
    lig_masked_path = f'/homes/hadfield/ZINC/RF_masking_store/lig{indices_test[positive_idx]}_masked.sdf'

    lig = Chem.MolFromMolFile(lig_path)
    prot = Chem.MolFromMolFile(prot_path)

    lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lig_path)][0]
    prot_oddt = [p for p in oddt.toolkit.readfile('sdf', prot_path)][0]


    #Compute PLEC fingerprint
    bits_info = {}
    fp = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, bits_info = bits_info, distance_cutoff = 4.5, size = 1024)
    fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, sparse = False, size = 1024)



    attribution_scores = []

    for atom_idx in range(lig.GetNumHeavyAtoms()):

        #print(atom_idx)

        #Get Attribution score
        attribution_score = get_masking_score(lig_path, prot_path, atom_idx, lig_masked_path)
        attribution_scores.append(attribution_score)

    #print(attribution_scores)

    atom_ranking = pd.DataFrame({'atom_idx':np.arange(lig.GetNumHeavyAtoms()), 'attribution':attribution_scores}).sort_values('attribution', ascending = False)
    #print(atom_ranking)
    atom_ranking.to_csv(f'/homes/hadfield/ZINC/RF_attribution_masking_store/atom_rankings_{positive_idx}.csv', index = False, sep = ' ')
    

    #Get coords of positive atoms
    positive_coords = assign_mol_label(lig, prot, threshold = 4)
    positive_coords_np = np.array(positive_coords)
    np.savetxt(f'/homes/hadfield/ZINC/RF_attribution_masking_store/positive_coords_{positive_idx}.txt', positive_coords_np)


    highest_ranking_positive = 0
    
    

    for idx, atom_id in enumerate(atom_ranking['atom_idx']):

        #print(atom_id, np.array(lig.GetConformer().GetAtomPosition(atom_id)))

        if checkArray(np.array(lig.GetConformer().GetAtomPosition(atom_id)), positive_coords):
            highest_ranking_positive = idx + 1
            break

    

    return highest_ranking_positive
'''

def get_masking_ranking_dataframe(lp_file_paths, RF_model, binding_threshold = 3.5, plec_threshold = 4.5, hydrophobic = False, delete_temp_file = True):
    
    
    lig_path = lp_file_paths[0]
    prot_path = lp_file_paths[1]
    
    #Create temporary directory to store
    lig_masked_dir = tempfile.mkdtemp()
    lig_masked_path = f'{lig_masked_dir}/lig_masked.sdf'

    #lig_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/ligands/lig{indices_test[positive_idx]}.sdf'
    #prot_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/pharmacophores/pharm{indices_test[positive_idx]}.sdf'
    #lig_masked_path = f'/homes/hadfield/ZINC/RF_masking_store/lig{indices_test[positive_idx]}_masked.sdf'

    lig = Chem.MolFromMolFile(lig_path)
    prot = Chem.MolFromMolFile(prot_path)

    #lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lig_path)][0]
    #prot_oddt = [p for p in oddt.toolkit.readfile('sdf', prot_path)][0]


    #Compute PLEC fingerprint
    #bits_info = {}
    #fp = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, bits_info = bits_info, distance_cutoff = 4.5, size = 1024)
    #fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, sparse = False, size = 1024)



    attribution_scores = []
    atom_coords = []
    
    for atom_idx in range(lig.GetNumHeavyAtoms()):

        #print(atom_idx)

        #Get Attribution score
        attribution_score = get_masking_score(lig_path, prot_path, atom_idx, lig_masked_path, RF_model, plec_threshold = plec_threshold)
        attribution_scores.append(attribution_score)
        
        pos = np.array(lig.GetConformer().GetAtomPosition(atom_idx))
        atom_coords.append(pos)
        
    #print(attribution_scores)
    
    atom_coords = np.array(atom_coords)
    


    atom_ranking = pd.DataFrame({'atom_idx':np.arange(lig.GetNumHeavyAtoms()), 'x':atom_coords[:, 0], 
                                 'y':atom_coords[:, 1], 'z':atom_coords[:, 2], 
                                 'attribution':attribution_scores}).sort_values('attribution', ascending = False)
    
    
    lig_label_df, pharm_label_df = create_gt_df(lig, prot, threshold=binding_threshold, hydrophobic = hydrophobic)
    
    #print(lig_label_df)
    #print(pharm_label_df)


    #Do index matching
    
    #atom_types = []
    involved_in_binding = []
    
    for idx, row in atom_ranking.iterrows():
        for jdx, sow in lig_label_df.iterrows():
            if vector_distance(np.array([row['x'], row['y'], row['z']]), np.array([sow['x'], sow['y'], sow['z']])) < 0.05:
                
                #atom_types.append(sow['type'])
                if not hydrophobic:
                    involved_in_binding.append(sow['binding'])
                else:
                    involved_in_binding.append(sow['contribution'])
    
    #atom_ranking['atom_type'] = atom_types
    atom_ranking['binding'] = involved_in_binding
    
    if delete_temp_file:
        shutil.rmtree(lig_masked_dir)

    return atom_ranking




if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')
    
    
    arguments = parser.parse_args()









'''
for idx, lab in enumerate(y_test):
    if idx < 20:
        if lab == 1: #positive example
            print(idx, get_highest_ranking_positive_masking(idx))
'''



'''

test_subset_positive = []

for idx, lab in enumerate(y_test):
    if idx < 5000:
        if lab == 1: #positive example
            test_subset_positive.append(idx) #append positive idx

pool = Pool(mp.cpu_count())

highest_ranking_positive_list = pool.map(get_highest_ranking_positive_masking, test_subset_positive)

out_df = pd.DataFrame({'idx':test_subset_positive, 'highest_ranking_positive':highest_ranking_positive_list})
out_df.to_csv('/homes/hadfield/ZINC/highest_ranking_positive_masking_df.csv', index = False, sep = ' ')


print('Job finished')


'''



'''
for idx, lab in enumerate(y_test):

    if idx < 10:
            
        if lab == 1: #positive example
            lig_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/ligands/lig{indices_test[idx]}.sdf'
            prot_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/pharmacophores/pharm{indices_test[idx]}.sdf'

            lig = Chem.MolFromMolFile(lig_path)
            prot = Chem.MolFromMolFile(prot_path)
            
            lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lig_path)][0]
            prot_oddt = [p for p in oddt.toolkit.readfile('sdf', prot_path)][0]


            #Compute PLEC fingerprint
            bits_info = {}
            fp = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, bits_info = bits_info, distance_cutoff = 4.5, size = 1024)
            fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, sparse = False, size = 1024)



            attribution_scores = []

            for atom_idx in range(lig.GetNumHeavyAtoms()):

                #print(atom_idx)
                features_of_interest = get_features_of_interest(atom_idx, lig, bits_info)
                #print('features of interest')
                #print(features_of_interest)

                #Get Attribution score
                attribution_score = attribution_score_for_features_and_example(X_test[idx], features_of_interest, clf, np.arange(X_test.shape[1]))
                attribution_scores.append(attribution_score)


            atom_ranking = pd.DataFrame({'atom_idx':np.arange(lig.GetNumHeavyAtoms()), 'attribution':attribution_scores}).sort_values('attribution', ascending = False)
            print(atom_ranking)
            

            #Get coords of positive atoms
            positive_coords = assign_mol_label(lig, prot)

            highest_ranking_positve = 0

            print('Positive coords:\n')
            for p in positive_coords:
                print(p)



            for idx, atom_id in enumerate(atom_ranking['atom_idx']):

                print(atom_id, np.array(lig.GetConformer().GetAtomPosition(atom_id)))

                if checkArray(np.array(lig.GetConformer().GetAtomPosition(atom_id)), positive_coords):
                    highest_ranking_positive = idx + 1
                    break

            print(highest_ranking_positive)
'''



'''


def get_highest_ranking_positive(positive_idx):

    lig_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/ligands/lig{indices_test[positive_idx]}.sdf'
    prot_path = f'/homes/hadfield/ZINC/zinc_50ops_ac0025_t4_data/sdf/pharmacophores/pharm{indices_test[positive_idx]}.sdf'

    lig = Chem.MolFromMolFile(lig_path)
    prot = Chem.MolFromMolFile(prot_path)

    lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lig_path)][0]
    prot_oddt = [p for p in oddt.toolkit.readfile('sdf', prot_path)][0]


    #Compute PLEC fingerprint
    bits_info = {}
    fp = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, bits_info = bits_info, distance_cutoff = 4.5, size = 1024)
    fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein = 0, sparse = False, size = 1024)



    attribution_scores = []

    for atom_idx in range(lig.GetNumHeavyAtoms()):

        #print(atom_idx)
        features_of_interest = get_features_of_interest(atom_idx, lig, bits_info)
        #print('features of interest')
        #print(features_of_interest)

        #Get Attribution score
        attribution_score = attribution_score_for_features_and_example(X_test[positive_idx], features_of_interest, clf, np.arange(X_test.shape[1]))
        attribution_scores.append(attribution_score)


    atom_ranking = pd.DataFrame({'atom_idx':np.arange(lig.GetNumHeavyAtoms()), 'attribution':attribution_scores}).sort_values('attribution', ascending = False)
t    #print(atom_ranking)
    atom_ranking.to_csv(f'/homes/hadfield/ZINC/RF_attribution_store/atom_rankings_{positive_idx}.csv', index = False, sep = ' ')
    

    #Get coords of positive atoms
    positive_coords = assign_mol_label(lig, prot, threshold = 4)
    positive_coords_np = np.array(positive_coords)
    np.savetxt(f'/homes/hadfield/ZINC/RF_attribution_store/positive_coords_{positive_idx}.txt', positive_coords_np)


    highest_ranking_positive = 0

    #print('Positive coords:\n')
    #for p in positive_coords:
        #print(p)



    for idx, atom_id in enumerate(atom_ranking['atom_idx']):

        #print(atom_id, np.array(lig.GetConformer().GetAtomPosition(atom_id)))

        if checkArray(np.array(lig.GetConformer().GetAtomPosition(atom_id)), positive_coords):
            highest_ranking_positive = idx + 1
            break

    return highest_ranking_positive



test_subset_positive = []


for idx, lab in enumerate(y_test):
    if idx < 2000:
        if lab == 1: #positive example
            test_subset_positive.append(idx) #append positive idx


pool = Pool(mp.cpu_count())

highest_ranking_positive_list = pool.map(get_highest_ranking_positive, test_subset_positive)

out_df = pd.DataFrame({'idx':test_subset_positive, 'highest_ranking_positive':highest_ranking_positive_list})
out_df.to_csv('/homes/hadfield/ZINC/highest_ranking_positive_df.csv', index = False, sep = ' ')


print('Job finished')
'''
