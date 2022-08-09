###################PROCESSING DATA FROM ZINC#################################
#cd ~/ZINC
#mkdir zinc_50ops_ac0025_t4_data

#python3 ~/SynthPharm/ds_generation/main.py  zinc_250k_confs_jpsettings_top1.sdf zinc_50ops_ac0025_t4_data -n 50 -a 0.025 -t 4.0 -mp


#python ~/DUDE/random_split.py zinc_50ops_ac0025_t4_data/parquets zinc_50ops_ac0025_t4_split


#cp  zinc_50ops_ac0025_t4_data/labels.yaml zinc_50ops_ac0025_t4_split
#cp  zinc_50ops_ac0025_t4_data/atomic_labels.yaml zinc_50ops_ac0025_t4_split


#cp  zinc_50ops_ac0025_t4_data/labels.yaml zinc_50ops_ac0025_t4_split/test
#cp  zinc_50ops_ac0025_t4_data/atomic_labels.yaml zinc_50ops_ac0025_t4_split/test

#cp  zinc_50ops_ac0025_t4_data/labels.yaml zinc_50ops_ac0025_t4_split/train

#tar -czvf ZINC_processed.tar.gz zinc_50ops_ac0025_t4_split


#############################################################################
#

input_SDF=$1
intermediate=$2
output_tar=$3


mkdir ${intermediate}_data

#python3 ~/SynthPharm/ds_generation/main.py  ${input_SDF} ${intermediate}_data -n 50 -a 0.025 -t 4.0 -mp

python ~/DUDE/random_split.py ${intermediate}_data/parquets ${intermediate}_split

cp ${intermediate}_data/labels.yaml ${intermediate}_split
cp ${intermediate}_data/atomic_labels.yaml ${intermediate}_split


cp ${intermediate}_data/labels.yaml ${intermediate}_split/test
cp ${intermediate}_data/atomic_labels.yaml ${intermediate}_split/test

cp ${intermediate}_data/labels.yaml ${intermediate}_split/train

tar -czvf ${output_tar} ${intermediate}_split



