root_dir=/data/hookbill/hadfield/syntheticVS/data/zinc_hydrophobic_forced_50ops_ac0025_st4_processed
st=4

#for fp in 2.5 3 3.5 4 4.5 5 5.5 6

#do
	
#	echo ${fp}

#	fp_str=$(echo $fp | sed 's/\.//')
#	output_plec=features_plec_${fp_str}
#	output_label=labels_plec_${fp_str}
#	python convert_to_numpy.py ${root_dir}/actives/sdf/ligands ${root_dir}/actives/sdf/pharmacophores ${root_dir}/actives -hy -s ${st} -of ${output_plec} -ol ${output_label} -p ${fp} 
#	python convert_to_numpy.py ${root_dir}/decoys/sdf/ligands ${root_dir}/decoys/sdf/pharmacophores ${root_dir}/decoys -hy -s ${st} -of ${output_plec} -ol ${output_label} -p ${fp}
#

#done



output_plec=features_morgan
output_label=labels_morgan
python convert_to_numpy.py ${root_dir}/actives/sdf/ligands ${root_dir}/actives/sdf/pharmacophores ${root_dir}/actives -hy -s ${st} -of ${output_plec} -ol ${output_label} -n
python convert_to_numpy.py ${root_dir}/decoys/sdf/ligands ${root_dir}/decoys/sdf/pharmacophores ${root_dir}/decoys -hy -s ${st} -of ${output_plec} -ol ${output_label} -n

