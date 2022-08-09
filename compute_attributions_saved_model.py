#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:02:18 2022

@author: hadfield
"""

import argparse

from RF_attribution_masking import get_masking_ranking_dataframe
import os
import joblib




def main(args):
    
    if not os.path.exists(args.sdf_ligand):
        return 0
    

    #load model
    clf = joblib.load(args.model_fname)

    if not os.path.exists(f'{args.root_dir_test}/{args.attribution_dir}'):
        os.mkdir(f'{args.root_dir_test}/{args.attribution_dir}')
      
    print(f'Calculating attribution dataset {args.save_idx}...')
    
    '''
    for idx, lp_path in enumerate(lp_paths):
        
        if idx % 100 == 0:
            print(f'Processing example {idx} of {len(lp_paths)}')
        
        out_df = get_masking_ranking_dataframe(lp_path,RF_model=clf_plec,binding_threshold=4)
        out_df.to_csv(f'{args.root_dir_test}/{args.attribution_dir}/df{idx}.csv', index = False, sep = ' ')
    '''
    out_df = get_masking_ranking_dataframe([args.sdf_ligand, args.sdf_pharm], RF_model = clf, 
                                           binding_threshold = args.binding_threshold, plec_threshold = args.plec_threshold, hydrophobic = args.hydrophobic)
    
    #out_df = get_masking_ranking_dataframe(lp_path,RF_model=clf_plec,binding_threshold=4, hydrophobic = True)

    
    out_df.to_csv(f'{args.root_dir_test}/{args.attribution_dir}/df{args.save_idx}.csv', index = False, sep = ' ')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir_test', type = str, help = 'Location of the molecules to be used for the test set')
    parser.add_argument('model_fname', type = str, help = 'Location of saved RF model')
    parser.add_argument('sdf_ligand', type = str, help = 'Location of ligand sdf file')
    parser.add_argument('sdf_pharm', type = str, help = 'Location of synthetic protein sdf file')
    
    parser.add_argument('--attribution_dir', '-ad', type = str, default = 'attribution_dfs', help = 'name of directory to store attribution dataset')
    parser.add_argument('--binding_threshold', '-bt', type = float, default = 4, help = 'Binding threshold for labelling')
    parser.add_argument('--plec_threshold', '-pt', type = float, default = 4.5, help = 'PLEC distance cutoff')
    parser.add_argument('--hydrophobic', '-hy', action='store_true', help = 'Hydrophobic dataset')

    parser.add_argument('--save_idx', '-s', type = int, default = 0, help = 'Index to save attribution dataframe in the attribution_dir')

    
    
    arguments = parser.parse_args()
    
    main(arguments)
