#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit RF model using PLEC/Morgan fingerprints

Created on Tue May 17 11:49:08 2022

@author: hadfield
"""


from rf_utils import *

import numpy as np
import argparse
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, average_precision_score

import joblib


def main(args):
    

    ###LOAD DATA IN TO TRAIN MODEL

    if not args.biased:

        if args.features_npy is None:
            X_plec = np.load(f'{args.root_dir}/features_plec.npy', allow_pickle = True)
        else:
            X_plec = np.load(f'{args.root_dir}/{args.features_npy}.npy', allow_pickle = True)
    
        if args.labels_npy is None:
            y = np.load(f'{args.root_dir}/labels.npy')
        else:
            y = np.load(f'{args.root_dir}/{args.labels_npy}.npy')
    
        #Import features and labels
        X_morgan = np.load(f'{args.root_dir}/features_morgan.npy', allow_pickle = True)
    else:
        
        X_actives_morgan = np.load(f'{args.root_dir}/actives/features_morgan.npy', allow_pickle = True)
        X_inactives_morgan = np.load(f'{args.root_dir}/decoys/features_morgan.npy', allow_pickle = True)
        
        
        X_actives_plec = np.load(f'{args.root_dir}/actives/{args.features_active_npy}.npy', allow_pickle = True)
        X_inactives_plec = np.load(f'{args.root_dir}/decoys/{args.features_decoy_npy}.npy', allow_pickle = True)
        
        y_actives = np.load(f'{args.root_dir}/actives/{args.labels_active_npy}.npy')
        y_inactives = np.load(f'{args.root_dir}/decoys/{args.labels_decoy_npy}.npy')
    
        y_actives = y_actives.reshape(-1)
        y_inactives = y_inactives.reshape(-1)
        
        #Stack on top of each other 
        X_morgan = np.vstack((X_actives_morgan, X_inactives_morgan))
        X_plec = np.vstack((X_actives_plec, X_inactives_plec))
        y = np.concatenate((y_actives, y_inactives), axis = 0)
    
    
    
    
    #Subset to get desired class balance if necessary:
    if args.prop_actives is not None:
        X_plec, y, X_morgan = control_prop_actives(X_plec, y, X_morgan, prop_actives = args.prop_actives)



    
    #Do train/test split
    
    X_morgan_train, X_morgan_test, X_plec_train, X_plec_test, y_train, y_test = tts(X_morgan, X_plec, y, test_size=args.test_set_proportion, random_state=42)
    
    #Perturb training set labels if necessary
    if args.noise_ratio is not None:
        y_train = perturb_labels(y_train, args.noise_ratio)


    ###FIT MODEL
    clf_morgan = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_morgan.fit(X_morgan_train, y_train)
    
    clf_plec = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_plec.fit(X_plec_train, y_train)
    
    if args.write_results:
        #Now assess the model performance on the test set
        pred_labels_morgan = clf_morgan.predict(X_morgan_test)
        pred_probs_morgan = clf_morgan.predict_proba(X_morgan_test)[:, 1]
    
        np.savetxt(f'{args.root_dir}/test_labels_true.txt', y_test)
        np.savetxt(f'{args.root_dir}/morgan_test_labels_pred.txt', pred_labels_morgan)
        np.savetxt(f'{args.root_dir}/morgan_test_labels_prob.txt', pred_probs_morgan)
    
    
    
        pred_labels_plec = clf_plec.predict(X_plec_test)
        pred_probs_plec = clf_plec.predict_proba(X_plec_test)[:, 1]
    
        np.savetxt(f'{args.root_dir}/plec_test_labels_pred.txt', pred_labels_plec)
        np.savetxt(f'{args.root_dir}/plec_test_labels_prob.txt', pred_probs_plec)
        
        
        #Accuracy 
        accuracy_morgan = accuracy_score(y_test, pred_labels_morgan)
        accuracy_plec = accuracy_score(y_test, pred_labels_plec)
        
        #ROC AUC
        roc_morgan = roc_auc_score(y_test, pred_probs_morgan)
        roc_plec = roc_auc_score(y_test, pred_probs_plec)
    
 
        #PRC AUC
        prc_morgan = average_precision_score(y_test, pred_probs_morgan)
        prc_plec = average_precision_score(y_test, pred_probs_plec)

        
        #log loss
        ll_morgan = log_loss(y_test, pred_probs_morgan)
        ll_plec = log_loss(y_test, pred_probs_plec)
    
    
        with open(f'{args.root_dir}/{args.morgan_output_fname}.txt', 'w') as f:
            f.write(f'Accuracy: {round(accuracy_morgan, 3)}\n')
            f.write(f'ROC AUC: {round(roc_morgan, 3)}\n')
            f.write(f'Logarithmic Loss: {round(ll_morgan, 3)}\n')
            f.write(f'PRC AUC: {round(prc_morgan, 3)}\n')

        with open(f'{args.root_dir}/{args.plec_output_fname}.txt', 'w') as f:
            f.write(f'Accuracy: {round(accuracy_plec, 3)}\n')
            f.write(f'ROC AUC: {round(roc_plec, 3)}\n')
            f.write(f'Logarithmic Loss: {round(ll_plec, 3)}\n')
            f.write(f'PRC AUC: {round(prc_plec, 3)}\n')


    if args.model_fname is not None:
        joblib.dump(clf_plec, args.model_fname, compress = 3)

    if args.model_morgan_fname is not None:
        joblib.dump(clf_morgan, args.model_morgan_fname, compress = 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('--test_set_proportion', '-t', type = int, default = 0.2, help = 'Specify size of test set')

    
    parser.add_argument('--noise_ratio', '-nr', type = float, default = None, help = 'Fraction of training examples to mislabel')
    parser.add_argument('--prop_actives', '-pa', type = float, default = None, help = 'Fraction of examples which should be actives (data will be subsetted)')
    
    parser.add_argument('--biased', '-b', action='store_true', help = 'Separate files for actives and decoys')
    
    parser.add_argument('--features_npy', '-f', type = str, default = None, help = 'npy array to train random forest - only used if data is not "biased"')
    parser.add_argument('--labels_npy', '-l', type = str, default = None, help = 'npy array to train random forest - only used if data is not "biased"' )

    parser.add_argument('--features_active_npy', '-fa', type = str, default = None, help = 'npy array of actives to train random forest - only used if data is "biased"')
    parser.add_argument('--labels_active_npy', '-la', type = str, default = None, help = 'npy array of actives to train random forest - only used if data is "biased"' )
    parser.add_argument('--features_decoy_npy', '-fd', type = str, default = None, help = 'npy array of decoys to train random forest - only used if data is "biased"')
    parser.add_argument('--labels_decoy_npy', '-ld', type = str, default = None, help = 'npy array of decoys to train random forest - only used if data is "biased"' )


    parser.add_argument('--write_results', '-w', action='store_true', help = 'Write model results on test set to file')
    parser.add_argument('--plec_output_fname', '-pof', type = str, default = 'plec_performance', help = 'Name for saving summary statistics')
    parser.add_argument('--morgan_output_fname', '-mof', type = str, default = 'morgan_performance', help = 'Name for saving summary statistics')

    parser.add_argument('--model_fname', type = str, default = None, help = 'Location to save trained model (caution: Random Forests can be very large)' )
    parser.add_argument('--model_morgan_fname', type = str, default = None, help = 'Location to save trained morgan fingerprint model (caution: Random Forests can be very large)' )



    arguments = parser.parse_args()
    
    main(arguments)
