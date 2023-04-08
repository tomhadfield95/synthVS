#EXAMPLE SCRIPT SHOWING HOW TO COMPUTRE ATTRIBUTIONS FOR CONTRIBUTION DATASETS WITH BIASED DATA STRUCTURE

#CHANGE THESE TO MATCH YOUR OWN DIRECTORIES
root_dir_test=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_hydrophobic_filtered_test_set
lig_root=${root_dir_test}/sdf/ligands/lig
pharm_root=${root_dir_test}/sdf/pharmacophores/pharm

pt=4 #PLEC distance cutoff

for dataset in DUDE_AA2AR DUDE_DRD3 DUDE_FA10 DUDE_MK14 DUDE_VGFR2 LIT_ALDH1 LIT_FEN1 LIT_MAPK1 LIT_PKM2 LIT_VDR; do
	echo Computing attributions for target ${dataset}
	pt_str=$(echo $pt | sed 's/\.//')
	
	#CHANGE TO MATCH YOUR OWN DIRECTORIES
	root_dir=/data/hookbill/hadfield/syntheticVS/data/${dataset}_hydrophobic_50ops_ac0025_st4_processed

	features_npy=features_plec
        labels_npy=labels
        results_name_plec=plec_performance_${dataset}_${pt_str}
        results_name_morgan=morgan_performance_${dataset}_${pt_str}
        model_fname=/tmp/rf_model_hydrophobic_plec_${dataset}_${pt_str}.joblib
        attr_dir=attributions_dir_${dataset}_${pt_str}


	#Fit model
	python ../../fit_rf.py ${root_dir} -fa ${features_npy} -fd ${features_npy} -la ${labels_npy} -ld ${labels_npy} --write_results -pof ${results_name_plec} -mof ${results_name_morgan} --model_fname ${model_fname} --biased
        

	#Compute Attributions
	parallel "python ../../compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt ${pt} -s {} -ad ${attr_dir} -hy" ::: {0..500}

done


