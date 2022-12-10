# SynthPharm
Ongoing project in collaboration with Jack Scantlebury, investigating the extent to which deep learning methods are able to correctly identify important functional groups in protein-ligand binding. TLDR: They're better than fingerprint-based methods, but only under the right circumstances. Manuscript to appear soon!

The only requirement for this project is
[PointVS](https://github.com/jscant/PointVS). A complete installation including
this requirement can be found below:

```
git clone git@github.com:jscant/PointVS
cd PointVS
conda env create -f environment.yml python=3.8
conda activate pointvs
pip install -e .
cd ..
git clone git@github.com:jscant/SynthPharm
cd SynthPharm
pip install -e . -vvv
```

## Dataset generation:
The script `ds_generation/main.py` takes the following arguments:

```
positional arguments:
  ligands               Location of ligand sdf(s)
  output_dir            Directory in which to store outputs

optional arguments:
  -h, --help            show this help message and exit
  --max_pharmacophores MAX_PHARMACOPHORES, -m MAX_PHARMACOPHORES
                        Maximum number of pharmacophores for each ligand
  --area_coef AREA_COEF, -a AREA_COEF
  --mean_pharmacophores MEAN_PHARMACOPHORES, -p MEAN_PHARMACOPHORES
                        Mean number of pharmacophores for each ligand
  --num_opportunities NUM_OPPORTUNITIES, -n NUM_OPPORTUNITIES
                        Number of interaction opportunities per ligand.
  --distance_threshold DISTANCE_THRESHOLD, -t DISTANCE_THRESHOLD
                        Maximum distance between ligand functional groups and
                        their respective pharmacophores for the combination of
                        the two to be considered an active
  --use_multiprocessing, -mp
                        Use multiple CPU processes
```

To run dataset generation on an SDF file containing multiple structures:
```
python3 ds_generation/main.py <INPUT.sdf> <OUTPUT_DIR> -n 50 -a 0.025 -t 4.0 -mp
```

This will produce a directory structure where in `<OUTPUT_DIR>` there are two
directories named `sdf` and `parquets`, each containing a further two 
directories called `ligands` and `pharmacophores`, which contain the original
ligand structures and the generated pharmacophores respectively, in either sdf
or parquet format. Two text files will also be produced in `<OUTPUT_DIR>`, which
are `labels.yaml` containing the filename index to label mapping, and some
statistics on the generated pharmacophores (`stats.txt`).
