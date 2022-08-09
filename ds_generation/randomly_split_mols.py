#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:38:32 2022

@author: hadfield
"""

import numpy as np
from rdkit import Chem

import sys

mols_path = sys.argv[1]
output_root_dir = sys.argv[2]
actives_fname = sys.argv[3]
decoys_fname = sys.argv[4]


mols = [x for x in Chem.SDMolSupplier(sys.argv[1])]
labels = np.random.binomial(1, 0.5, len(mols))

w = Chem.SDWriter(f'{output_root_dir}/{actives_fname}.sdf')
v = Chem.SDWriter(f'{output_root_dir}/{decoys_fname}.sdf')


for idx, m in enumerate(mols):
    if labels[idx] == 1:
        w.write(m) #Write as active
    else:
        v.write(m) #Write as decoy
        


