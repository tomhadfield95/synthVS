#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:17:32 2022

@author: hadfield
"""

import argparse
import glob
import pandas as pd
import numpy as np
from point_vs.utils import mkdir, save_yaml

def check_pharm_mol(pharm_path):
    
    df = pd.read_parquet(pharm_path)
    if df.shape[0] > 0:
        return True
    else:
        return False
    

def main(args):
    
    if args.forced:
        
        for t in ['test', 'train']:
            mkdir(args.output_dir, t, 'ligands')
            mkdir(args.output_dir, t, 'pharmacophores')
            

        
        labels = {}
        all_pharms = glob.glob(f'{args.root_dir}/*/parquets/pharmacophores/*.parquet')
        print('{args.root_dir}/*/parquets/pharmacophores/*.parquet')
        print(len(all_pharms))
        idx = 0
    
        for pharm_file in all_pharms:
            
            if idx % 5000 == 0:
                print(idx)
            
            pharm_pq = pd.read_parquet(pharm_file)
            
            if pharm_pq.shape[0] > 0:
                input_idx = pharm_file.split('pharm')[2].split('.')[0]
                
                if 'actives' in pharm_file:
                    matching_lig = f'{args.root_dir}/actives/parquets/ligands/lig{input_idx}.parquet'
                    labels[idx] = 1
                elif 'decoys' in pharm_file:
                    matching_lig = f'{args.root_dir}/decoys/parquets/ligands/lig{input_idx}.parquet'
                    labels[idx] = 0
                
                lig_pq = pd.read_parquet(matching_lig)
                
                
                samp = np.random.binomial(1, 0.8)
                if samp == 0:
                    loc = 'test'
                else:
                    loc = 'train'
                
                lig_pq.to_parquet(f'{args.output_dir}/{loc}/ligands/lig{idx}.parquet')
                pharm_pq.to_parquet(f'{args.output_dir}/{loc}/pharmacophores/pharm{idx}.parquet')
                
                idx+=1
    
        save_yaml(labels, f'{args.output_dir}/labels.yaml')
        save_yaml(labels, f'{args.output_dir}/train/labels.yaml')
        save_yaml(labels, f'{args.output_dir}/test/labels.yaml')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory')
    parser.add_argument('output_dir', type = str, help = 'Location of output directory')

    parser.add_argument('--forced', '-f', action = 'store_true',
                        help = "The labels of the input data have been 'forced' - i.e. there are separate 'active' and 'decoy' subdirectories ")
    
    parser.add_argument
    

    arguments = parser.parse_args()

    main(arguments)
                                                                                                 