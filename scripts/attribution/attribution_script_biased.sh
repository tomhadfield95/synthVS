#Train model


root_dir=/data/hookbill/hadfield/syntheticVS/data
root_dir_test=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set
lig_root=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set/sdf/ligands/lig
pharm_root=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set/sdf/pharmacophores/pharm

for target in AA2AR DRD3 FA10 MK14 VGFR2;

do
	
	echo ${target} 4
	model_fname=/tmp/rf_model_DUDE_${target}.joblib
	attr_dir=attributions_dir_DUDE_${target}_4_balanced
	python fit_rf.py ${root_dir}/DUDE_${target}_50ops_ac0025_t4_processed -fa features_plec -fd features_plec -la labels -ld labels --write_results -pof plec_performance_balanced_4 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
	parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 4 -s {} -ad ${attr_dir}" ::: {0..500}

	

	echo ${target} 4.5
        model_fname=/tmp/rf_model_DUDE_${target}.joblib
        attr_dir=attributions_dir_DUDE_${target}_45_balanced
	python fit_rf.py ${root_dir}/DUDE_${target}_50ops_ac0025_t4_processed -fa features_plec_45 -fd features_plec_45 -la labels_45 -ld labels_45 --write_results -pof plec_performance_balanced_45 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 4.5 -s {} -ad ${attr_dir}" ::: {0..500}



	echo ${target} 5
        model_fname=/tmp/rf_model_DUDE_${target}.joblib
        attr_dir=attributions_dir_DUDE_${target}_5_balanced
        python fit_rf.py ${root_dir}/DUDE_${target}_50ops_ac0025_t4_processed -fa features_plec_5 -fd features_plec_5 -la labels_5 -ld labels_5 --write_results -pof plec_performance_balanced_5 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 5 -s {} -ad ${attr_dir}" ::: {0..500}






done


for target in ALDH1 FEN1 MAPK1 PKM2 VDR; do

	#echo ${target}
        #model_fname=/tmp/rf_model_LIT_${target}.joblib
        #attr_dir=attributions_dir_LIT_${target}_45_balanced
        #python fit_rf.py ${root_dir}/LIT_${target}_50ops_ac0025_t4_processed -fa features_plec -fd features_plec -la labels -ld labels --write_results -pof plec_performance_balanced_45 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        #parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 5 -s {} -ad ${attr_dir}" ::: {0..500}


	echo ${target} 4
        model_fname=/tmp/rf_model_LIT_${target}.joblib
        attr_dir=attributions_dir_LIT_${target}_4_balanced
        python fit_rf.py ${root_dir}/LIT_${target}_50ops_ac0025_t4_processed -fa features_plec -fd features_plec -la labels -ld labels --write_results -pof plec_performance_balanced_4 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 4 -s {} -ad ${attr_dir}" ::: {0..500}



        echo ${target} 4.5
        model_fname=/tmp/rf_model_LIT_${target}.joblib
        attr_dir=attributions_dir_LIT_${target}_45_balanced
        python fit_rf.py ${root_dir}/LIT_${target}_50ops_ac0025_t4_processed -fa features_plec_45 -fd features_plec_45 -la labels_45 -ld labels_45 --write_results -pof plec_performance_balanced_45 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 4.5 -s {} -ad ${attr_dir}" ::: {0..500}



        echo ${target} 5
        model_fname=/tmp/rf_model_LIT_${target}.joblib
        attr_dir=attributions_dir_LIT_${target}_5_balanced
        python fit_rf.py ${root_dir}/LIT_${target}_50ops_ac0025_t4_processed -fa features_plec_5 -fd features_plec_5 -la labels_5 -ld labels_5 --write_results -pof plec_performance_balanced_5 -mof morgan_performance_balanced --model_fname ${model_fname} --biased -pa 0.5
        parallel "python compute_attributions_saved_model.py ${root_dir_test} ${model_fname} ${lig_root}{}.sdf ${pharm_root}{}.sdf -pt 5 -s {} -ad ${attr_dir}" ::: {0..500}





done






