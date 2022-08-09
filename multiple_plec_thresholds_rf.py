#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:53:03 2022

@author: hadfield
"""

import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


root_dir = '/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed'

for plec_threshold in [25, 35, 3, 45, 4, 55, 5, 6]:
#for plec_threshold in [45]: 
   
    X = np.load(f'{root_dir}/features_plec_{plec_threshold}.npy')
    y = np.load(f'{root_dir}/labels_plec_{plec_threshold}.npy')
    
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
    
    clf_plec = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_plec.fit(X_train, y_train)

    pred_labels_plec = clf_plec.predict(X_test)
    pred_probs_plec = clf_plec.predict_proba(X_test)[:, 1]

    np.savetxt(f'{root_dir}/plec_test_labels_pred_{plec_threshold}.txt', pred_labels_plec)
    np.savetxt(f'{root_dir}/plec_test_labels_prob_{plec_threshold}.txt', pred_probs_plec)


    #Accuracy 
    accuracy_plec = accuracy_score(y_test, pred_labels_plec)

    #ROC AUC
    roc_plec = roc_auc_score(y_test, pred_probs_plec)


    #log loss
    ll_plec = log_loss(y_test, pred_probs_plec)


    
    with open(f'{root_dir}/plec_performance_{plec_threshold}.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy_plec, 3)}\n')
        f.write(f'ROC AUC: {round(roc_plec, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll_plec, 3)}\n')
