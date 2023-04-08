#Example script showing how to train a random forest model for generated synthetic data

root_dir=<root_directory_for_generated_data> #If training on "biased data" this should contain the "actives" and "decoys" subdirectories. 
					     #Otherwise it should contain the .npy files containing the molecular fingerprints used to train the RF
#NOTE: If training on "biased" data, you must include the "--biased" (or "-b") flag in the python command

python ../../fit_rf.py --features_npy features_plec.npy --labels_npy labels.npy --write_results


