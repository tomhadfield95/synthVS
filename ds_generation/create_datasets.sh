####ZINC####

#ligands=/data/hookbill/hadfield/syntheticVS/data/zinc_10k.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -mp


####DUDE SRC####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_SRC.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_SRC_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_SRC.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_SRC_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


#####LIT PCBA####

ligands=/data/hookbill/hadfield/syntheticVS/data/actives_ALDH1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_ALDH1_50ops_ac0025_t4_processed/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_ALDH1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_ALDH1_50ops_ac0025_t4_processed/decoys
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


ligands=/data/hookbill/hadfield/syntheticVS/data/actives_FEN1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_FEN1_50ops_ac0025_t4_processed/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_FEN1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_FEN1_50ops_ac0025_t4_processed/decoys
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


ligands=/data/hookbill/hadfield/syntheticVS/data/actives_MAPK1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_MAPK1_50ops_ac0025_t4_processed/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_MAPK1.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_MAPK1_50ops_ac0025_t4_processed/decoys
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


ligands=/data/hookbill/hadfield/syntheticVS/data/actives_PKM2.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_PKM2_50ops_ac0025_t4_processed/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_PKM2.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_PKM2_50ops_ac0025_t4_processed/decoys
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


ligands=/data/hookbill/hadfield/syntheticVS/data/actives_VDR.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_VDR_50ops_ac0025_t4_processed/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_VDR.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_VDR_50ops_ac0025_t4_processed/decoys
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp















################################################

#ligands=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_filtered_test_set.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/pdbbind_small_hydrophobic_filtered_test_set
#num_ops=50
#ac=0.025
#dist_threshold=4
#score_threshold=4




#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 1 -mp
#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 1 -mp -hy -st ${score_threshold}

####DUDE SRC####
#ligands=/data/hookbill/hadfield/syntheticVS/data/aa2ar/actives_final.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_AA2AR_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/aa2ar/decoys_final.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_AA2AR_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


####ZINC SIMPLIFIED####

#ligands=/data/hookbill/hadfield/syntheticVS/data/zinc_10k.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_simplified_50ops_ac0025_t6_forced_plec6_processed
#num_ops=50
#ac=0.025
#dist_threshold=6

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -mp -sl -f 1




####ZINC Only one pharmacophore####

#ligands=/data/hookbill/hadfield/syntheticVS/data/zinc_10k.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_simplified_1pharm_t6_forced_plec6_processed
#dist_threshold=6

#python main.py ${ligands} ${output_dir} -m 1 -p 1 -t ${dist_threshold} -mp -sl -f 1






####DUDE AA2AR####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_AA2AR.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_AA2AR_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_AA2AR.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_AA2AR_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp



####DUDE FA10####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_FA10.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_FA10_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_FA10.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_FA10_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


####DUDE VGFR2####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_VGFR2.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_VGFR2_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_VGFR2.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_VGFR2_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


####DUDE MK14####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_MK14.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_MK14_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_MK14.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_MK14_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp


####DUDE DRD3####
ligands=/data/hookbill/hadfield/syntheticVS/data/actives_DRD3.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_DRD3_50ops_ac0025_t4_processed_2/actives
num_ops=50
ac=0.025
dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_DRD3.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_DRD3_50ops_ac0025_t4_processed_2/decoys
num_ops=50
ac=0.025
dist_threshold=4

python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp




####DUDE FA10####
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_DUDE_all.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_ALL_50ops_ac0025_t4_processed/actives
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp

#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_DUDE_all.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_ALL_50ops_ac0025_t4_processed/decoys
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -f 2 -mp







