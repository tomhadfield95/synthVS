target1=$1
root_dir_1=/data/hookbill/hadfield/syntheticVS/data/DUDE_${target1}_50ops_ac0025_t4_processed



python compute_attributions_biased.py /data/hookbill/hadfield/syntheticVS/data/DUDE_AA2AR_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_DUDE_AA2AR_pc45_balanced -pf features_plec_45 -b

python compute_attributions_biased.py /data/hookbill/hadfield/syntheticVS/data/DUDE_DRD3_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_DUDE_DRD3_pc45_balanced -pf features_plec_45 -b

python compute_attributions_biased.py /data/hookbill/hadfield/syntheticVS/data/DUDE_FA10_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_DUDE_FA10_pc45_balanced -pf features_plec_45 -b

python compute_attributions_biased.py /data/hookbill/hadfield/syntheticVS/data/DUDE_MK14_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_DUDE_MK14_pc45_balanced -pf features_plec_45 -b

python compute_attributions_biased.py /data/hookbill/hadfield/syntheticVS/data/DUDE_VGFR2_50ops_ac0025_t4_processed /data/hookbill/hadfield/syntheticVS/data/pdbbind_filtered_test_set -ad attribution_dfs_DUDE_VGFR2_pc45_balanced -pf features_plec_45 -b

