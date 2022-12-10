#Train model


root_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed/
root_dir_test=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set
lig_root=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set/sdf/ligands/lig
pharm_root=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set/sdf/pharmacophores/pharm


#python fit_rf.py ${root_dir} -f ${features_npy} -l ${labels_npy} --write_results -pof ${results_name_plec} -mof ${results_name_morgan} --model_fname ${model_fname}
#parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt ${pt} -s {} -ad ${attr_dir}  " ::: {0..500}


features_npy=features_morgan
labels_npy=labels
model_fname=/tmp/rf_model_morgan.joblib
attr_dir=attributions_dir_morgan


#python fit_rf.py ${root_dir} -f ${features_npy} -l ${labels_npy}  --model_fname ${model_fname}
parallel "python compute_attributions_morgan.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -s {} -ad ${attr_dir}  " ::: {0..500}


