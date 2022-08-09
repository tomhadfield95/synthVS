import argparse
import numpy as np

from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import pickle
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

def main(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    #Import features and labels
    X = np.load(args.features, allow_pickle = True)
    y = np.load(args.labels, allow_pickle = True)

    X_train, X_test, y_train, y_test = tts(X, y, random_state = 0)

    print(X_train.shape, y_train.shape)
    #example = X_train[0]
    #print(np.array(example))
    #arr = np.zeros((0,), dtype=np.int8)
    #DataStructs.ConvertToNumpyArray(example,arr)
    #print(arr)
    
    if args.regression:
        clf = rfr(n_estimators = 1000, random_state = 0, n_jobs = -1)
        clf.fit(X_train, y_train)
    else:
        clf = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
        clf.fit(X_train, y_train)


    if args.save_model is not None:
        with open(f'{args.output_dir}/{args.save_model}.pickle', 'wb') as f:
            pickle.dump(clf, f)
    

    #Now assess the model performance on the test set
    pred_labels = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)[:, 1]
    
    
    if args.save_predictions:
        np.savetxt(f'{args.output_dir}/test_labels_true.txt', y_test)
        np.savetxt(f'{args.output_dir}/test_labels_pred.txt', pred_labels)
        np.savetxt(f'{args.output_dir}/test_labels_prob.txt', pred_probs)
    
   
    
    #Accuracy 
    accuracy = accuracy_score(y_test, pred_labels)
    
    #ROC AUC
    roc = roc_auc_score(y_test, pred_probs)
    
    #log loss
    ll = log_loss(y_test, pred_probs)
    
    
    with open(f'{args.output_dir}/model_performance.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy, 3)}\n')
        f.write(f'ROC AUC: {round(roc, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll, 3)}\n')
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=str, help='Numpy array of features')
    parser.add_argument('labels', type = str, help = 'Numpy array of labels')
    parser.add_argument('output_dir', type=str,
                        help='Location to save output')
    parser.add_argument('--regression', '-r', action = 'store_true',
                        help = 'Fit a regression model instead of a classication one')
    parser.add_argument('--save_model', '-s', type = str, default = None,
                        help = 'If None, the model will not be saved, otherwise will be saved in the output_dir with the name provided in this argument')
    parser.add_argument('--save_predictions', '-p', action = 'store_true',
                        help = 'Save the true test labels and the ones predicted by the model')
    
    

    arguments = parser.parse_args()
    
    main(arguments)