ligands=<insert_ligand_sdfs_filepath_here>
output_dir=<insert_output_directory_here>

#Example parameters
num_ops=50
ac=0.025
dist_threshold=4

python ../../main.py ${ligands} ${output_dir} -n ${num_ops} -a ${ac} -t ${dist_threshold} -mp
