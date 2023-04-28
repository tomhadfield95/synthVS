#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:16:54 2022

@author: hadfield
"""

from label import assign_mol_label

from rdkit import Chem
import glob
import numpy as np
import pandas as pd
from utils import save_yaml
import argparse

def main(args):
    
    ligands = glob.glob(f'{args.root_dir}/sdf/ligands/*.sdf')
    out_dict = {}
    
    for lig_file in ligands:
        
        idx = int(lig_file.split('lig')[2].split('.')[0])
        pharm_file = f'{args.root_dir}/sdf/pharmacophores/pharm{idx}.sdf'
        
        lig_mol = Chem.MolFromMolFile(lig_file)
        pharm_mol = Chem.MolFromMolFile(pharm_file)
        
        _, _, lig_df, _ = assign_mol_label(lig_mol, pharm_mol, hydrophobic=True, return_contrib_df=True)
        
        out_dict[idx] = lig_df.values
    
    save_yaml(out_dict, f'{args.root_dir}/atomic_labels.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory')

        
    arguments = parser.parse_args()

    main(arguments)
