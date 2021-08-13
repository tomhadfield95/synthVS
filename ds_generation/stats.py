import argparse
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from point_vs.utils import expand_path, load_yaml, pretify_dict
from rdkit import Chem

from filters import get_pharm_numbers
from generate import define_box_around_ligand


def simple_summary_stats(mols, pharm_mols, labels):
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

    df = pd.DataFrame(
        {'mols': mols, 'pharm_mols': pharm_mols, 'labels': labels})
    print('Loaded dataframe')

    mol_stats = namedtuple(
        'mol_stats',
        'ligand_atoms pharm_atoms ligand_donors ligand_acceptors '
        'pharm_box_ratio')

    df_actives = df.loc[df['labels'] == 1]
    df_inactives = df.loc[df['labels'] == 0]

    # Average number of polar atoms
    donor_count_active, acceptor_count_active = get_donor_acceptor_numbers(
        df_actives['mols'])
    donor_count_inactive, acceptor_count_inactive = get_donor_acceptor_numbers(
        df_inactives['mols'])
    print('Counted polar atoms')

    # Check pharmacophore/box area ratio
    active_ratio = get_pharm_box_ratio(df_actives)
    inactive_ratio = get_pharm_box_ratio(df_inactives)
    print('Checked pharmacophore/box area ratio')

    active_stats = mol_stats(
        ligand_atoms=round(float(np.mean(donor_count_active)), 3),
        pharm_atoms=round(float(np.mean(
            [m.GetNumHeavyAtoms() for m in df_actives["pharm_mols"]])), 3),
        ligand_donors=round(float(np.mean(donor_count_active)), 3),
        ligand_acceptors=round(float(np.mean(acceptor_count_active)), 3),
        pharm_box_ratio=round(float(np.mean(active_ratio)), 3)
    )

    inactive_stats = mol_stats(
        ligand_atoms=round(float(np.mean(donor_count_inactive)), 3),
        pharm_atoms=round(float(np.mean(
            [m.GetNumHeavyAtoms() for m in df_inactives["pharm_mols"]])), 3),
        ligand_donors=round(float(np.mean(donor_count_inactive)), 3),
        ligand_acceptors=round(float(np.mean(acceptor_count_inactive)), 3),
        pharm_box_ratio=round(float(np.mean(inactive_ratio)), 3)
    )

    return active_stats, inactive_stats


def write_statistics(fname, lig_mols, pharm_mols, label_dict):
    sorted_labels = [label for _, label in sorted(
        label_dict.items(), key=lambda x: x[0])]
    active_stats, inactive_stats = simple_summary_stats(
        mols=lig_mols, pharm_mols=pharm_mols, labels=sorted_labels)

    stats_str = 'Active_stats:\n'
    stats_str += pretify_dict(active_stats._asdict())
    stats_str += 'Inactive stats:'
    stats_str += pretify_dict(inactive_stats._asdict())
    with open(fname, 'w') as f:
        f.write(stats_str + '\n')
    return stats_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sdf_root', type=str,
                        help='Location of two directories named ligands and '
                             'pharmacophores in which sdf output of '
                             'ds_generation/main.py are stored.')
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

    print(write_statistics(
        sdf_root.parent / 'stats.txt', lig_mols, pharm_mols, labels))
