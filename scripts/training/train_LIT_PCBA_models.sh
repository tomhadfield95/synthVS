#python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py /data/hookbill/hadfield/syntheticVS/data/LIT_ALDH1_50ops_ac0025_t4_processed -pf features_plec_45 -ssp plec_performance_45 -b
#python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py /data/hookbill/hadfield/syntheticVS/data/LIT_FEN1_50ops_ac0025_t4_processed -pf features_plec_45 -ssp plec_performance_balanced -ssm morgan_performance_balanced -b
python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py /data/hookbill/hadfield/syntheticVS/data/LIT_MAPK1_50ops_ac0025_t4_processed -pf features_plec_45 -ssp plec_performance_45_balanced -b -ssm morgan_performance_balanced 
#python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py /data/hookbill/hadfield/syntheticVS/data/LIT_PKM2_50ops_ac0025_t4_processed -pf features_plec_45 -ssp plec_performance_45
#python /data/hookbill/hadfield/SynthPharm/fit_rf_biased.py /data/hookbill/hadfield/syntheticVS/data/LIT_VDR_50ops_ac0025_t4_processed -pf features_plec_45 -ssp plec_performance_45

