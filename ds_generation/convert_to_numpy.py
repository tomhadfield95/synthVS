#Script to convert the generated ligands/pharmacophores into a numpy array 
#So that we can train the random forest on

import argparse

import numpy as np
import multiprocessing as mp
import glob
import os


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from label import assign_mol_label

import oddt 
from oddt import fingerprints
import json

def get_fingerprint_morgan(lig_path):
    
    lig = Chem.MolFromMolFile(lig_path)
    if lig is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(lig, 2, nBits=1024)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    else:
        return None


def get_fingerprint_plec(lp_pair, distance_cutoff = 4):
    
    #Check if None:
    lig = Chem.MolFromMolFile(lp_pair[0])
    prot = Chem.MolFromMolFile(lp_pair[1])

    if lig is None or prot is None:
        return None
    else:


        lig_oddt = [l for l in oddt.toolkit.readfile('sdf', lp_pair[0])][0]
        prot_oddt = [l for l in oddt.toolkit.readfile('sdf', lp_pair[1])][0]


        fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein=0, sparse = False, size = 1024, distance_cutoff = distance_cutoff) 

        return fp_dense


def get_fingerprint_plec_mod_cutoff(arguments):

    #Alternative implementation where the PLEC distance cutoff is included as an argument in the vector so that we can change it and still do multiprocessing

    #arguments
    #arguments[0] - the path to the ligand sdf
    #arguments[1] - the path to the protein sdf
    #arguments[2] - the PLEC distance cutoff

    lig_oddt = [l for l in oddt.toolkit.readfile('sdf', arguments[0])][0]
    prot_oddt = [l for l in oddt.toolkit.readfile('sdf', arguments[1])][0]

    fp_dense = fingerprints.PLEC(lig_oddt, prot_oddt, depth_ligand = 3, depth_protein=0, sparse = False, size = 1024, distance_cutoff = arguments[2])
    
    return fp_dense


def get_label(lig, prot, hydrophobic = False, score_threshold = None, distance_threshold = None):
    
    #lig, prot are both rdkit molecules representing the ligand and protein score respectively
    #hydrophobic controls whether use a distance threshold from a matching pharmacophore to label
        #or whether we assign each interaction a score and add them up.
    
    
    if not hydrophobic:
        #We're using the distance threshold cutoff to label
        
        positive_coords = assign_mol_label(lig, prot, threshold = distance_threshold)
        return int(len(positive_coords) > 0)
        
    else:
        
        interaction_score = assign_mol_label(lig, prot, hydrophobic = True)
        
        if score_threshold is None:
            return interaction_score
        else:
            return int(interaction_score > score_threshold)
        
















def main(args):
    
    
    pool = mp.Pool(mp.cpu_count()-1) #multiprocessing
    lig_list = sorted(glob.glob(f'{args.ligands}/*.sdf'))
    pharm_list = sorted(glob.glob(f'{args.pharmacophores}/*.sdf'))

    num_examples = len(lig_list)
    example_indices = [int(x.split('lig')[-1].split('.')[0]) for x in lig_list]

    print(f'Length of lig_list, pre-filtering: {num_examples}')


    if args.labels_dict is not None:
        with open(args.labels_dict, 'r') as f:
            labels = json.load(f)
        




    #Get features
    
    if args.labels_dict is not None:
        calc_plec_args = []
        labels_list = []
        lig_list_filtered = []    

        #for idx in range(num_examples):
        for idx in example_indices:
            if f'{args.ligands}/lig{idx}.sdf' in lig_list and f'{args.pharmacophores}/pharm{idx}.sdf' in pharm_list and str(idx) in list(labels.keys()):
                calc_plec_args.append([f'{args.ligands}/lig{idx}.sdf', f'{args.pharmacophores}/pharm{idx}.sdf', args.plec_threshold])
                labels_list.append(labels[str(idx)])
                lig_list_filtered.append(f'{args.ligands}/lig{idx}.sdf')
    else:
        calc_plec_args = []
        lig_list_filtered = []
        #for idx in range(num_examples):
        for idx in example_indices:
            if f'{args.ligands}/lig{idx}.sdf' in lig_list and f'{args.pharmacophores}/pharm{idx}.sdf' in pharm_list:
                calc_plec_args.append([f'{args.ligands}/lig{idx}.sdf', f'{args.pharmacophores}/pharm{idx}.sdf', args.plec_threshold])
                lig_list_filtered.append(f'{args.ligands}/lig{idx}.sdf')
            else: 
                print(f'{args.ligands}/lig{idx}.sdf', f'{args.pharmacophores}/pharm{idx}.sdf')

    print(f'Length of lig_list, post-filtering: {len(lig_list_filtered)}')


    if not args.no_pharmacophores:
        
        #lp_pair = [[l, pharm_list[i]] for i, l in enumerate(lig_list)]
        #features = pool.map(get_fingerprint_plec, lp_pair)

        features = pool.map(get_fingerprint_plec_mod_cutoff, calc_plec_args)
    
    else:
            
        features = pool.map(get_fingerprint_morgan, lig_list_filtered)
    
    features = np.array(features)

    
    if arguments.labels_dict is None:
        
        #If we haven't provided labels then compute them now
        #Get Labels
        labels_list = []
        for idx, lp_paths in enumerate(calc_plec_args):
            lig = Chem.MolFromMolFile(lp_paths[0])
            prot = Chem.MolFromMolFile(lp_paths[1])
            
            label = get_label(lig, prot, args.hydrophobic, args.score_threshold, args.distance_threshold)
            labels_list.append(label)
            
        labels_list = np.array(labels_list)
        
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    #Save features and labels
    np.save(f'{args.output_dir}/{args.output_name_features}.npy', features)
    np.save(f'{args.output_dir}/{args.output_name_labels}.npy', labels_list)
    
    
    return 0


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands', type=str, help='Directory containing all ligand sdf(s)')
    parser.add_argument('pharmacophores', type = str, help = 'Directory containing all pharmacophore sdf(s)')
    parser.add_argument('output_dir', type=str,
                        help='Location to save the .npy file representing the features')
    parser.add_argument('--output_name_features', '-of', default = 'features',
                        help = 'name of the .npy file (not including the extension) where the features will be saved')
    parser.add_argument('--output_name_labels', '-ol', default = 'labels',
                        help = 'name of the .npy file (not including the extension) where the labels will be saved')
    
    parser.add_argument('--hydrophobic', '-hy', action = 'store_true')
    parser.add_argument('--score_threshold', '-s', type = float, default = None,
                        help = 'Cutoff value for labelling an example as positive when using the hydrophobic setup')
    parser.add_argument('--distance_threshold', '-d', type = float, default = None,
                        help = 'Cutoff value for labelling an example as positive when using the polar-only setup')

    parser.add_argument('--plec_threshold', '-p', type = float, default = 4.0,
                        help = 'Cutoff value for computing the PLEC fingerprint')

    parser.add_argument('--no_pharmacophores', '-n', action = 'store_true',
                        help = 'Generate Morgan FP using only the ligands')
    
    parser.add_argument('--labels_dict', '-ld', type = str, default = None, 
                        help = 'Location of json file containing the labels')


    arguments = parser.parse_args()
    
    main(arguments)
