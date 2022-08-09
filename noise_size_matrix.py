#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:37:57 2021

@author: hadfield
"""

import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

import json

def trainModel(trainingFeatures, trainingLabels, regression = False):

    #trainingFeatures - a numpy array containing the training features
                    #- a ligand/protein pair is featurised using PLEC fingerprints

    #trainingLabels - a numpy array containing the training labels
                   #- should be 0/1 if regression==False and floats if regression==True

    #saveModelName - name of the model to be saved. If not specified the trained model will be saved as 'RF_model_{date and time}.pickle'

    #regression - Boolean describing if the model will be a regression or classification model

    #Define the model
    if regression:
        clf = rfr(n_estimators = 1000, random_state = 0, n_jobs = -1)
        clf.fit(trainingFeatures, trainingLabels)
    else:
        clf = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
        clf.fit(trainingFeatures, trainingLabels)

    
    return clf


def perturb_labels(labs, p = 0.05):

    perturbed_labs = []
    for lab in labs:
        perturb = np.random.binomial(1, p)
        if perturb == 1:
            perturbed_labs.append(1 - lab)
        else:
            perturbed_labs.append(lab)
    return np.array(perturbed_labs)




#get data 
perturb_prob = [0, 1, 2, 5, 10, 20, 30, 40, 50]
#training_size = [100, 1000, 5000, 10000, 20000, 50000, 100000, 150000]
#training_size = np.arange(20, 1000, 40)
training_size = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]


X = np.load('/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed/features_plec_45.npy')
y = np.load('/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed/labels_plec_45.npy')


#X_hydrophobic = np.load('/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_50ops_ac0025_st4_processed/features_plec.npy')
#y_hydrophobic = np.load('/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_50ops_ac0025_st4_processed/labels.npy')



X_train, X_test, y_train, y_test = tts(X, y, random_state = 0)
#X_train_hy, X_test_hy, y_train_hy, y_test_hy = tts(X_hydrophobic, y_hydrophobic, random_state = 0)





results_matrix = {}
results_matrix_hy = {}

for n in training_size:
    results_matrix[str(n)] = {}
    #results_matrix_hy[str(n)] = {}
    print(f'Training models of size {n}')

    for p in perturb_prob:

        results_matrix[str(n)][str(p)] = {}
        #results_matrix_hy[str(n)][str(p)] = {}

        X_subset = X_train[:n, :]
        y_subset = y_train[:n]
        y_subset = perturb_labels(y_subset, p/100)
        

        #X_subset_hy = X_train_hy[:n, :]
        #y_subset_hy = y_train_hy[:n]
        #y_subset_hy = perturb_labels(y_subset_hy, p/100)



        
        #Fit model
        clf = trainModel(X_subset, y_subset)
        #clf_hy = trainModel(X_subset_hy, y_subset_hy)



        #Predict on test set
        #Now assess the model performance on the test set
        pred_labels = clf.predict(X_test)
        pred_probs = clf.predict_proba(X_test)[:, 1]
        
        #pred_labels_hy = clf_hy.predict(X_test_hy)
        #pred_probs_hy = clf_hy.predict_proba(X_test_hy)[:, 1]


        accuracy = accuracy_score(y_test, pred_labels)
        roc = roc_auc_score(y_test, pred_probs)
        logloss= log_loss(y_test, pred_probs)

        #accuracy_hy = accuracy_score(y_test_hy, pred_labels_hy)
        #roc_hy = roc_auc_score(y_test_hy, pred_probs_hy)
        #logloss_hy = log_loss(y_test_hy, pred_probs_hy)


        results_matrix[str(n)][str(p)]['accuracy'] = accuracy
        results_matrix[str(n)][str(p)]['roc'] = roc
        results_matrix[str(n)][str(p)]['logloss'] = logloss

        #results_matrix_hy[str(n)][str(p)]['accuracy'] = accuracy_hy
        #results_matrix_hy[str(n)][str(p)]['roc'] = roc_hy
        #results_matrix_hy[str(n)][str(p)]['logloss'] = logloss_hy


        print(f'Accuracy for training set size {n} and noise level {p/100}: {round(accuracy, 3)}')
        #print(f'Accuracy for training set size {n} and noise level {p/100} on the hydrophobic set: {round(accuracy_hy, 3)}')

#with open(f'/data/hookbill/hadfield/syntheticVS/zinc_50ops_ac0025_t4_data/noise_size_matrix_small.json', 'w') as mat:
    #json.dump(results_matrix, mat)

with open('/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed/noise_size_matrix_small.json', 'w') as mat:
    json.dump(results_matrix, mat)

#with open('/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_50ops_ac0025_st4_processed/noise_size_matrix_small.json', 'w') as mat:
    #json.dump(results_matrix_hy, mat)






