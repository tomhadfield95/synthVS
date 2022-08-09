#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF Util functions

Created on Tue May 17 11:50:31 2022

@author: hadfield
"""

import pandas as pd
import numpy as np



def control_prop_actives(features, labels, additional_features = None, prop_actives = None):
    
    #additional_features is included in case we have to do the same subsetting with 
    #a different type of feature set - e.g. a Morgan Fingerprint.
    
    if prop_actives is None:
        return features, labels #i.e. don't change the proportion at all
    elif prop_actives < 0 or prop_actives > 1:
            raise ValueError('Please provide a proportion between 0 and 1.')
    else:
        
        initial_num_actives = sum(labels == 1)
        initial_num_inactives = sum(labels == 0)
        
        if additional_features is None:
            df = pd.DataFrame({'labels':labels, 'features':[x for x in features]})
        else:
            df = pd.DataFrame({'labels':labels, 'features':[x for x in features], 'add_features':[x for x in additional_features]})
            
            
        df_inactives = df.loc[df['labels'] == 0]
        df_actives = df.loc[df['labels'] == 1]

        if initial_num_actives*(1/prop_actives - 1) < initial_num_inactives:
            #We can use all the existing actives and just subset the inactives
            num_to_sample = initial_num_actives * (1/prop_actives - 1)
            df_inactives_subset = df_inactives.sample(n = int(np.floor(num_to_sample)), random_state = 100)
            df_combined_subset = df_actives.append(df_inactives_subset)
        
        else:
            num_to_sample = initial_num_inactives*(prop_actives/(1 - prop_actives))
            df_actives_subset = df_actives.sample(n = int(np.floor(num_to_sample)), random_state = 100)
            df_combined_subset = df_inactives.append(df_actives_subset)
            
            
        #shuffle order
        df_combined_subset = df_combined_subset.sample(frac=1)
        subset_labels = np.array(df_combined_subset['labels'])
        subset_features = np.array(list(df_combined_subset['features']))
        subset_additional_features = np.array(list(df_combined_subset['add_features']))
       
        
        print(f'Proportion actives: {np.mean(subset_labels)}')
        if additional_features is None:
            return subset_features, subset_labels
        else:
            return subset_features, subset_labels, subset_additional_features
            


def perturb_labels(labs, p = 0.05):

    perturbed_labs = []
    for lab in labs:
        perturb = np.random.binomial(1, p)
        if perturb == 1:
            perturbed_labs.append(1 - lab)
        else:
            perturbed_labs.append(lab)
    return np.array(perturbed_labs)

        
