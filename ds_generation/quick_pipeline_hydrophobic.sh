
#data=/data/hookbill/hadfield/syntheticVS/data/zinc_10k.sdf
#output=/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_50ops_ac0025_st4_processed

#do dataset generation
#python main.py ${data} ${output} -a 0.025 -n 50 -t 4  -mp -hy -st 4

#train model
#python /data/hookbill/hadfield/SynthPharm/fit_rf_unbiased.py ${output} -t 1000




#echo 'Performance on Morgan Fingerprints'
#cat ${output}/morgan_performance.txt

#echo '*********************'

#echo 'Performance on PLEC Fingerprints'
#cat ${output}/plec_performance.txt


#get result


##FORCED LABELLING


data_active=/data/hookbill/hadfield/syntheticVS/data/actives_ZINC.sdf
data_decoy=/data/hookbill/hadfield/syntheticVS/data/decoys_ZINC.sdf

output_root=/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_forced_50ops_ac0025_st4_processed
output_actives=${output_root}/actives
output_decoys=${output_root}/decoys


#data_active=~/HDD_OPIG/syntheticVS_store/zinc_250k_actives.sdf
#data_decoy=~/HDD_OPIG/syntheticVS_store/zinc_250k_decoys.sdf


#output_root=/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_forced_50ops_ac0025_st4_processed
#output_root=~/HDD_OPIG/syntheticVS_store/zinc_hydrophobic_forced_250k_50ops_ac0025_st4_processed
#output_actives=${output_root}/actives
#output_decoys=${output_root}/decoys



num_ops=50
ac=0.025
score_threshold=4

#python main.py ${data_active} ${output_actives} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp
#python main.py ${data_decoy} ${output_decoys} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp

python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py ${output_root}

echo 'Performance on Morgan Fingerprints'
cat ${output_root}/morgan_performance.txt

echo '*********************'

echo 'Performance on PLEC Fingerprints'
cat ${output_root}/plec_performance.txt

