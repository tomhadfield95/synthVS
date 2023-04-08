#Example of how to convert the generated SDF files to numpy arrays which can be provided to an sklearn model
#This example is for a "biased" dataset, i.e. the actives and inactives are stored separately (as we force the labels when generating them)
#To do this conversion for an "unbiased" dataset, i.e. where the actives and inactives were randomly generated at the same time, we only require a single python command below - pointing to the ligands and synthetic proteins respectively


root_dir=<insert_directory_with_actives_and_decoys_as_subdirectories>

#Parameters

binding_distance_cutoff=4 #Distance for Polar Deterministic Binding Rule
			  #Can also use Contribution rule, but need to use -hy and -s parameters instead of -d

plec_fingerprint_cutoff=4.5
fp_str=$(echo ${plec_fingerprint_cutoff} | sed 's/\.//') #Removes decimal if one provided
output_plec=features_plec_${fp_str}
output_label=labels_plec_${fp_str}

python ../../ds_generation/convert_to_numpy.py ${root_dir}/actives/sdf/ligands ${root_dir}/actives/sdf/pharmacophores ${root_dir}/actives -d ${binding_distance_cutoff} -of ${output_plec} -ol ${output_label} -p ${plec_fingerprint_cutoff}
python ../../ds_generation/convert_to_numpy.py ${root_dir}/decoys/sdf/ligands ${root_dir}/decoys/sdf/pharmacophores ${root_dir}/decoys -d ${binding_distance_cutoff} -of ${output_plec} -ol ${output_label} -p ${plec_fingerprint_cutoff}

