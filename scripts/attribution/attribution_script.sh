#EXAMPLE OF RUNNING ATTRIBUTION FOR ZINC_POLAR DATA ON PDBBIND TEST SET WITH ALL DIFFERENT PLEC FINGERPRINT CUTOFFS

#Change to match your own directories
root_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed/
root_dir_test=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set

lig_root=${root_dir_test}/sdf/ligands/lig
pharm_root=${root_dir_test}/sdf/ligands/lig


for pt in 2.5 3 3.5 4 4.5 5 5.5 6; do
  echo Computing attributions for PLEC cutoff $pt
  pt_str=$(echo $pt | sed 's/\.//')
	

  	features_npy=features_plec_${pt_str}
	labels_npy=labels_plec_${pt_str}
	results_name_plec=plec_performance_${pt_str}
	results_name_morgan=morgan_performance_${pt_str}
	model_fname=/tmp/rf_model_plec_${pt_str}.joblib
	attr_dir=attributions_dir_${pt_str}



	python ../../fit_rf.py ${root_dir} -f ${features_npy} -l ${labels_npy} --write_results -pof ${results_name_plec} -mof ${results_name_morgan} --model_fname ${model_fname}
	parallel "python ../../compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt ${pt} -s {} -ad ${attr_dir}  " ::: {0..500} #500 is number of examples in test set - change to suit your purposes

done 



