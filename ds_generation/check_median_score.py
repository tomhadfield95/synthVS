#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:10:00 2021

@author: hadfield
"""

import numpy as np
import json
import argparse

def main(args):
    
    with open(args.labels, 'r') as f:
        labels_dict = json.load(f)
        
    label_keys = sorted([int(k) for k in labels_dict.keys()])

    labels = np.array([labels_dict[f'{k}'] for k in label_keys])
    
    print(np.median(labels))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str, help='json file containing labels')
    arguments = parser.parse_args()
    
    main(arguments)
