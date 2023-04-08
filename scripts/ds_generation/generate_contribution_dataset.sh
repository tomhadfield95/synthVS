ligands=<insert_ligand_sdfs_filepath_here>
output_dir=<insert_output_directory_here>

#do dataset generation - -hy flag includes hydrophobic synthetic residues - i.e. contribution dataset
python ../../ds_generation/main.py ${ligands} ${output_dir} -a 0.025 -n 50  -mp -hy -st 4


