####ZINC####

#ligands=/data/hookbill/hadfield/syntheticVS/data/zinc_10k.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_50ops_ac0025_t4_processed
#num_ops=50
#ac=0.025
#dist_threshold=4

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -mp


num_ops=50
ac=0.025
score_threshold=4


#####LIT PCBA####

#names='ALDH1 FEN1 MAPK1 PKM2 VDR'
names='MAPK1 PKM2'
for name in $names
do

echo $name
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_${name}.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_${name}_hydrophobic_50ops_ac0025_st4_processed/actives
#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp


ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_${name}.sdf
output_dir=/data/hookbill/hadfield/syntheticVS/data/LIT_${name}_hydrophobic_50ops_ac0025_st4_processed/decoys

python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp

#mv /data/hookbill/hadfield/syntheticVS/data/LIT_${name}_hydrophobic_50ops_ac0025_t4_processed/decoys /data/hookbill/hadfield/syntheticVS/data/LIT_${name}_hydrophobic_50ops_ac0025_st4_processed/decoys

#scp -r /data/hookbill/hadfield/syntheticVS/data/LIT_${name}_hydrophobic_50ops_ac0025_st4_processed/ pegasus:/data/pegasus/hadfield/syntheticVS_hydrophobic



done
echo All done


######DUD-E######
names='AA2AR DRD3 FA10 MK14 VGFR2'
for name in $names
do

echo $name
#ligands=/data/hookbill/hadfield/syntheticVS/data/actives_${name}.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_${name}_hydrophobic_50ops_ac0025_st4_processed/actives
#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp


#ligands=/data/hookbill/hadfield/syntheticVS/data/decoys_${name}.sdf
#output_dir=/data/hookbill/hadfield/syntheticVS/data/DUDE_${name}_hydrophobic_50ops_ac0025_t4_processed/decoys

#python main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -hy -st ${score_threshold} -f 2 -mp

#mv /data/hookbill/hadfield/syntheticVS/data/DUDE_${name}_hydrophobic_50ops_ac0025_t4_processed/decoys /data/hookbill/hadfield/syntheticVS/data/DUDE_${name}_hydrophobic_50ops_ac0025_st4_processed/decoys

#scp -r /data/hookbill/hadfield/syntheticVS/data/DUDE_${name}_hydrophobic_50ops_ac0025_st4_processed/ pegasus:/data/pegasus/hadfield/syntheticVS_hydrophobic

done
echo All done



