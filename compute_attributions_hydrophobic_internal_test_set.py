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

def main(args):
 

    #Import features and labels
    X_plec_train = np.load(f'{args.root_dir}/train/sdf/features_plec.npy', allow_pickle = True)

    y_train = np.load(f'{args.root_dir}/train/sdf/labels_plec.npy')
    y_train = y_train.reshape(-1)

   
    #Fit random forest
   
    clf_plec = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
    clf_plec.fit(X_plec_train, y_train)
  
  
    all_lig_paths = sorted(glob.glob(f'{args.root_dir}/test/sdf/ligands/lig*.sdf'))
    all_pharm_paths = sorted(glob.glob(f'{args.root_dir}/test/sdf/pharmacophores/pharm*.sdf'))
    
    
    lp_paths = [[all_lig_paths[i], all_pharm_paths[i]] for i in range(len(all_lig_paths))]
    
    if not os.path.exists(f'{args.root_dir}/{args.attribution_dir}'):
        os.mkdir(f'{args.root_dir}/{args.attribution_dir}')
    
    
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
        
        out_df = get_masking_ranking_dataframe(lp_path,RF_model=clf_plec,binding_threshold=4, hydrophobic = True)
        out_df.to_csv(f'{args.root_dir}/{args.attribution_dir}/df{idx}.csv', index = False, sep = ' ')
    
    
    #test_lig_path = f'{args.root_dir_test}/sdf/ligands/lig1.sdf'
    #test_pharm_path = f'{args.root_dir_test}/sdf/pharmacophores/pharm1.sdf'
    
    #test_out_df = get_masking_ranking_dataframe([test_lig_path, test_pharm_path],RF_model=clf_plec,binding_threshold=4, hydrophobic = True)
    



    #print(test_out_df)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str, help = 'Location of root directory where numpy arrays are stored')
    parser.add_argument('--attribution_dir', '-ad', type = str, default = 'attribution_dfs', help = 'name of directory to store attribution datasets')


    parser.add_argument('--test_set_size', '-t', type = int, default = 500, help = 'Specify size of test set')
    
    
    arguments = parser.parse_args()
    
    main(arguments)
