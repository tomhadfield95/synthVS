import argparse
import multiprocessing as mp
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from point_vs.utils import expand_path, load_yaml, pretify_dict
from rdkit import Chem

from filters import get_pharm_numbers
from generate import define_box_around_ligand


def mean_of_named_tuples(tups):
    """Merge a list of named tuples, such that the mean is taken of each
    value"""

    def merge_dicts(house_d, brick_d):
        for key in house_d.keys():
            house_d[key] += brick_d[key]

    init_dict = tups[0]._asdict()
    for tup in tups[1:]:
        merge_dicts(init_dict, tup._asdict())
    return tups[0].__class__(
        **{key: value / len(tups) for key, value in init_dict.items()})


def simple_summary_stats(mols, pharm_mols, labels, cpus=1):
    def get_pharm_box_ratio(df):
        ratios = []
        for idx, row in df.iterrows():
            conf = row['mols'].GetConformer()
            positions = np.array(
                [np.array(conf.GetAtomPosition(i)) for i in
                 range(row['mols'].GetNumAtoms())])
            box_params = define_box_around_ligand(positions)
            box_area = (box_params['X'][1] - box_params['X'][0]) * \
                       (box_params['Y'][1] - box_params['Y'][0]) * (
                               box_params['Z'][1] - box_params['Z'][0])
            ratios.append(row['pharm_mols'].GetNumHeavyAtoms() / box_area)
        return ratios

    def get_donor_acceptor_numbers(df):
        donor_counts_active, acceptor_counts_active = [], []
        for m in df['mols']:
            pharm_counts = get_pharm_numbers(m)
            donor_counts = pharm_counts['Donor']
            acceptor_counts = pharm_counts['Acceptor']
            donor_counts_active.append(donor_counts)
            acceptor_counts_active.append(acceptor_counts)
        return donor_counts_active, acceptor_counts_active

    def process_mols(mol_df):
        df_actives = mol_df.loc[mol_df['labels'] == 1]
        df_inactives = mol_df.loc[mol_df['labels'] == 0]

        # Average number of polar atoms
        donor_count_active, acceptor_count_active = get_donor_acceptor_numbers(
            df_actives)
        donor_count_inactive, acceptor_count_inactive = \
            get_donor_acceptor_numbers(df_inactives)

        # Check pharmacophore/box area ratio
        active_ratio = get_pharm_box_ratio(df_actives)
        inactive_ratio = get_pharm_box_ratio(df_inactives)

        active_stats = mol_stats(
            ligand_atoms=round(float(np.mean(
                [m.GetNumHeavyAtoms() for m in df_actives["mols"]])), 3),
            pharm_atoms=round(float(np.mean(
                [m.GetNumHeavyAtoms() for m in df_actives["pharm_mols"]])), 3),
            ligand_donors=round(float(np.mean(donor_count_active)), 3),
            ligand_acceptors=round(float(np.mean(acceptor_count_active)), 3),
            pharm_box_ratio=round(float(np.mean(active_ratio)), 3)
        )

        inactive_stats = mol_stats(
            ligand_atoms=round(float(np.mean(
                [m.GetNumHeavyAtoms() for m in df_inactives["mols"]])), 3),
            pharm_atoms=round(float(np.mean(
                [m.GetNumHeavyAtoms() for m in df_inactives["pharm_mols"]])),
                3),
            ligand_donors=round(float(np.mean(donor_count_inactive)), 3),
            ligand_acceptors=round(float(np.mean(acceptor_count_inactive)), 3),
            pharm_box_ratio=round(float(np.mean(inactive_ratio)), 3)
        )

        return active_stats, inactive_stats

    df = pd.DataFrame(
        {'mols': mols, 'pharm_mols': pharm_mols, 'labels': labels})
    print('Loaded dataframe')

    mol_stats = namedtuple(
        'mol_stats',
        'ligand_atoms pharm_atoms ligand_donors ligand_acceptors '
        'pharm_box_ratio')

    if cpus > 1:
        dfs = np.array_split(df, mp.cpu_count())
        results = Pool().map(process_mols, dfs)
        active_stats = [r[0] for r in results]
        inactive_stats = [r[1] for r in results]
        return mean_of_named_tuples(
            active_stats), mean_of_named_tuples(inactive_stats)
    else:
        return process_mols(df)


def write_statistics(fname, lig_mols, pharm_mols, label_dict, cpus=1,
                     args_dict=None):
    def _round(x, decimals):
        return '{{:.{}f}}'.format(decimals).format(x)

    sorted_labels = [label for _, label in sorted(
        label_dict.items(), key=lambda x: x[0])]
    active_stats, inactive_stats = simple_summary_stats(
        mols=lig_mols, pharm_mols=pharm_mols, labels=sorted_labels, cpus=cpus)

    if isinstance(args_dict, dict):
        stats_str = 'Program arguments:\n'
        stats_str += pretify_dict(args_dict) + '\n\n'
    else:
        stats_str = ''
    stats_str += 'Active_stats:\n'
    stats_str += pretify_dict(
        {i: _round(j, 5) for i, j in active_stats._asdict().items()})
    stats_str += '\n\nInactive stats:\n'
    stats_str += pretify_dict(
        {i: _round(j, 5) for i, j in inactive_stats._asdict().items()})
    with open(fname, 'w') as f:
        f.write(stats_str + '\n')
    return stats_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sdf_root', type=str,
                        help='Location of two directories named ligands and '
                             'pharmacophores in which sdf output of '
                             'ds_generation/main.py are stored.')
    parser.add_argument('multiprocessing', action='store_true',
                        help='Use multiple cpus')
    args = parser.parse_args()

    sdf_root = expand_path(args.sdf_root)
    lig_sdfs = [str(p) for p in sorted(
        list(sdf_root.glob('ligands/*.sdf')),
        key=lambda x: int(Path(x.name).stem[3:]))]
    pharm_sdfs = [str(p) for p in sorted(
        list(sdf_root.glob('pharmacophores/*.sdf')),
        key=lambda x: int(Path(x.name).stem[5:]))]
    labels = load_yaml(sdf_root.parent / 'labels.yaml')

    assert len(pharm_sdfs) == len(lig_sdfs), (
        'Number of pharmacophores files must equal the number of ligand files')
    assert len(labels) == len(lig_sdfs), (
        'Number of labels must equal the number of input files')

    lig_mols = [Chem.SDMolSupplier(sdf_file)[0] for sdf_file in lig_sdfs]
    pharm_mols = [Chem.SDMolSupplier(sdf_file)[0] for sdf_file in pharm_sdfs]

    cpus = mp.cpu_count() if args.multiprocessing else 1
    print(write_statistics(
        sdf_root.parent / 'stats.txt', lig_mols, pharm_mols, labels, cpus=cpus))
