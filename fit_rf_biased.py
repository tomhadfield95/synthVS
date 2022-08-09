import numpy as np
import argparse
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, average_precision_score

import os


def main(args):
    

    #Import features and labels
    X_actives_morgan = np.load(f'{args.root_dir}/actives/{args.morgan_features_output}.npy', allow_pickle = True)
    X_inactives_morgan = np.load(f'{args.root_dir}/decoys/{args.morgan_features_output}.npy', allow_pickle = True)
    
    
    X_actives_plec = np.load(f'{args.root_dir}/actives/{args.plec_features_output}.npy', allow_pickle = True)
    X_inactives_plec = np.load(f'{args.root_dir}/decoys/{args.plec_features_output}.npy', allow_pickle = True)
    
    y_actives = np.load(f'{args.root_dir}/actives/{args.label_output}.npy')
    y_inactives = np.load(f'{args.root_dir}/decoys/{args.label_output}.npy')

    y_actives = y_actives.reshape(-1)
    y_inactives = y_inactives.reshape(-1)
    
    if args.force_balance:
        num_actives = y_actives.shape[0]
        num_inactives = y_inactives.shape[0]

        if num_actives < num_inactives:
            X_inactives_morgan = X_inactives_morgan[:num_actives]
            X_inactives_plec = X_inactives_plec[:num_actives]
            y_inactives = y_inactives[:num_actives]



    #Stack on top of each other 
    
    X_morgan = np.vstack((X_actives_morgan, X_inactives_morgan))
    X_plec = np.vstack((X_actives_plec, X_inactives_plec))
    y = np.concatenate((y_actives, y_inactives), axis = 0)


    #Do train/test split
    test_prop = args.test_set_size/y.shape[0]
    
    X_morgan_train, X_morgan_test, X_plec_train, X_plec_test, y_train, y_test = tts(X_morgan, X_plec, y, test_size=test_prop, random_state=42)
    
    #Fit random forest
    
    clf_morgan = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_morgan.fit(X_morgan_train, y_train)
    
    clf_plec = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_plec.fit(X_plec_train, y_train)
    
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


    with open(f'{args.root_dir}/{args.sum_stats_morgan}.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy_morgan, 3)}\n')
        f.write(f'ROC AUC: {round(roc_morgan, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll_morgan, 3)}\n')
        f.write(f'PRC AUC: {round(prc_morgan, 3)}\n')

    with open(f'{args.root_dir}/{args.sum_stats_plec}.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy_plec, 3)}\n')
        f.write(f'ROC AUC: {round(roc_plec, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll_plec, 3)}\n')
        f.write(f'PRC AUC: {round(prc_plec, 3)}\n')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')
    parser.add_argument('--force_balance', '-b', action = "store_true", help = 'Subset inactives to be the same size as the actives')   
        
    parser.add_argument('--plec_features_output', '-pf', default = 'features_plec', help = 'Name of plec fingerprint file')
    parser.add_argument('--morgan_features_output', '-mf', default = 'features_morgan', help = 'Name of morgan fingerprint file')
    parser.add_argument('--label_output', '-l', default = 'labels', help = 'Name of labels file')

    parser.add_argument('--sum_stats_morgan', '-ssm', default = 'morgan_performance', help = 'Name of txt file to write summary stats for the ligand-based model')
    parser.add_argument('--sum_stats_plec', '-ssp', default = 'plec_performance', help = 'Name of txt file to write summary stats for the structure-based model')


    arguments = parser.parse_args()
    
    main(arguments)
