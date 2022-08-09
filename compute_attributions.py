import numpy as np
import argparse
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from RF_attribution_masking import get_masking_ranking_dataframe
import glob
import os
from functools import partial
import multiprocessing as mp

def perturb_labels(labs, p = 0.05):

    perturbed_labs = []
    for lab in labs:
        perturb = np.random.binomial(1, p)
        if perturb == 1:
            perturbed_labs.append(1 - lab)
        else:
            perturbed_labs.append(lab)
    return np.array(perturbed_labs)


def main(args):
    

    #Import features and labels
    
    if args.features_npy is None:
        X_plec = np.load(f'{args.root_dir}/features_plec.npy', allow_pickle = True)
    else:
        X_plec = np.load(f'{args.root_dir}/{args.features_npy}.npy', allow_pickle = True)

    
    if args.labels_npy is None:
        y = np.load(f'{args.root_dir}/labels.npy')
    else: 
        y = np.load(f'{args.root_dir}/{args.labels_npy}.npy')
    
    #Do train/test split
    test_prop = args.test_set_size/y.shape[0]
    
    X_plec_train, X_plec_test, y_train, y_test = tts(X_plec, y, test_size=test_prop, random_state=42)
    

    if args.noise_ratio is not None:
        y_train = perturb_labels(y_train, args.noise_ratio)






    #Fit random forest
    
    #clf_morgan = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    #clf_morgan.fit(X_morgan_train, y_train)
    
    clf_plec = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_plec.fit(X_plec_train, y_train)
    
    #Now assess the model performance on the test set
    #pred_labels_morgan = clf_morgan.predict(X_morgan_test)
    #pred_probs_morgan = clf_morgan.predict_proba(X_morgan_test)[:, 1]

    #np.savetxt(f'{args.root_dir}/test_labels_true.txt', y_test)
    #np.savetxt(f'{args.root_dir}/morgan_test_labels_pred.txt', pred_labels_morgan)
    #np.savetxt(f'{args.root_dir}/morgan_test_labels_prob.txt', pred_probs_morgan)



    #pred_labels_plec = clf_plec.predict(X_plec_test)
    #pred_probs_plec = clf_plec.predict_proba(X_plec_test)[:, 1]

    #np.savetxt(f'{args.root_dir}/plec_test_labels_pred.txt', pred_labels_plec)
    #np.savetxt(f'{args.root_dir}/plec_test_labels_prob.txt', pred_probs_plec)
    
    
    #Accuracy 
    #accuracy_morgan = accuracy_score(y_test, pred_labels_morgan)
    #accuracy_plec = accuracy_score(y_test, pred_labels_plec)
    
    #ROC AUC
    #roc_morgan = roc_auc_score(y_test, pred_probs_morgan)
    #roc_plec = roc_auc_score(y_test, pred_probs_plec)


    #log loss
    #ll_morgan = log_loss(y_test, pred_probs_morgan)
    #ll_plec = log_loss(y_test, pred_probs_plec)


    #with open(f'{args.root_dir}/morgan_performance.txt', 'w') as f:
        #f.write(f'Accuracy: {round(accuracy_morgan, 3)}\n')
        #f.write(f'Logarithmic Loss: {round(ll_morgan, 3)}\n')
        
    #with open(f'{args.root_dir}/plec_performance.txt', 'w') as f:
        #f.write(f'Accuracy: {round(accuracy_plec, 3)}\n')
        #f.write(f'ROC AUC: {round(roc_plec, 3)}\n')
        #f.write(f'Logarithmic Loss: {round(ll_plec, 3)}\n')


    #Now we've fit the models we can do the attribution
    
    all_lig_paths = sorted(glob.glob(f'{args.root_dir_test}/sdf/ligands/lig*.sdf'))[:300]
    all_pharm_paths = sorted(glob.glob(f'{args.root_dir_test}/sdf/pharmacophores/pharm*.sdf'))[:300]
    
    
    lp_paths = [[all_lig_paths[i], all_pharm_paths[i]] for i in range(len(all_lig_paths))]
    
    if not os.path.exists(f'{args.root_dir_test}/{args.attribution_dir}'):
        os.mkdir(f'{args.root_dir_test}/{args.attribution_dir}')
    
    
    #def masking_single_argument(lp_path):
        #return get_masking_ranking_dataframe(lp_path,RF_model=clf_plec,binding_threshold=4)
    


    #masking_partial = partial(get_masking_ranking_dataframe, RF_model = clf_plec, binding_threshold = 4)
    #pool = mp.Pool(mp.cpu_count() - 1)
    
    print('Calculating attribution datasets...')
    
    #out_dfs = pool.map(masking_partial, lp_paths)
    #out_dfs = pool.map(masking_single_argument, lp_paths)
    #for idx, df in enumerate(out_dfs):
        #df.to_csv(f'{args.root_dir_test}/{args.attribution_dir}/df{idx}.csv')
    
    
    
    for idx, lp_path in enumerate(lp_paths):
        
        if idx % 100 == 0:
            print(f'Processing example {idx} of {len(lp_paths)}')
        
        out_df = get_masking_ranking_dataframe(lp_path,RF_model=clf_plec,binding_threshold=4)
        out_df.to_csv(f'{args.root_dir_test}/{args.attribution_dir}/df{idx}.csv', index = False, sep = ' ')
    
    
    #test_lig_path = f'{args.root_dir_test}/sdf/ligands/lig1.sdf'
    #test_pharm_path = f'{args.root_dir_test}/sdf/pharmacophores/pharm1.sdf'
    
    #test_out_df = get_masking_ranking_dataframe([test_lig_path, test_pharm_path],RF_model=clf_plec,binding_threshold=4)
    



    #print(test_out_df)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('root_dir_test', type = str, help = 'Location of the molecules to be used for the test set')
    parser.add_argument('--attribution_dir', '-ad', type = str, default = 'attribution_dfs', help = 'name of directory to store attribution datasets')

    parser.add_argument('--features_npy', '-f', type = str, default = None, help = 'npy array to train random forest')
    parser.add_argument('--labels_npy', '-l', type = str, default = None, help = 'npy array to train random forest')

    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')
    parser.add_argument('--noise_ratio', '-nr', type = float, default = None, help = 'Fraction of training examples to mislabel')   
    
    arguments = parser.parse_args()
    
    main(arguments)
