noise_ratio=$1
name=$2

python compute_attributions.py /data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_${name} -f features_plec_4 -l labels_plec_4 -nr ${noise_ratio} 


