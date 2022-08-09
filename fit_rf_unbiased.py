import numpy as np
import argparse
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


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
       
    
        if additional_features is None:
            return subset_features, subset_labels
        else:
            return subset_features, subset_labels, additional_features
            


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
    

    '''

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


    '''

    
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
    
    
    
    #Subset to get desired class balance if necessary:
    
    if args.prop_actives is not None:
        X_plec, y, X_morgan = control_prop_actives(X_plec, y, X_morgan, prop_actives = args.prop_actives)



    
    #Do train/test split
    test_prop = args.test_set_size/y.shape[0]
    
    X_morgan_train, X_morgan_test, X_plec_train, X_plec_test, y_train, y_test = tts(X_morgan, X_plec, y, test_size=test_prop, random_state=42)
    
    #Perturb training set labels if necessary
    if args.noise_ratio is not None:
        y_train = perturb_labels(y_train, args.noise_ratio)

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


    #log loss
    ll_morgan = log_loss(y_test, pred_probs_morgan)
    ll_plec = log_loss(y_test, pred_probs_plec)


    with open(f'{args.root_dir}/{args.morgan_output_fname}.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy_morgan, 3)}\n')
        f.write(f'ROC AUC: {round(roc_morgan, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll_morgan, 3)}\n')
        
    with open(f'{args.root_dir}/{args.plec_output_fname}.txt', 'w') as f:
        f.write(f'Accuracy: {round(accuracy_plec, 3)}\n')
        f.write(f'ROC AUC: {round(roc_plec, 3)}\n')
        f.write(f'Logarithmic Loss: {round(ll_plec, 3)}\n')


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')

    parser.add_argument('--features_npy', '-f', type = str, default = None, help = 'npy array to train random forest')
    parser.add_argument('--labels_npy', '-l', type = str, default = None, help = 'npy array to train random forest')
    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')
    parser.add_argument('--noise_ratio', '-nr', type = float, default = None, help = 'Fraction of training examples to mislabel')
    parser.add_argument('--prop_actives', '-pa', type = float, default = None, help = 'Fraction of examples which should be actives (data will be subsetted)')

    parser.add_argument('--plec_output_fname', '-pof', type = str, default = 'plec_performance', help = 'Name for saving summary statistics')
    parser.add_argument('--morgan_output_fname', '-mof', type = str, default - 'morgan_performance', help = 'Name for saving summary statistics')


    arguments = parser.parse_args()
    
    main(arguments)
