#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic pharmacophores. Code written by, or modified from code
written by, Tom Hadfield.
"""
import argparse
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from point_vs.utils import expand_path, mkdir, save_yaml, pretify_dict, \
    format_time, Timer
from rdkit import Chem

from filters import sample_from_pharmacophores, \
    pharm_pharm_distance_filter, pharm_ligand_distance_filter
from generate import create_pharmacophore_mol
from label import assign_mol_label
from stats import write_statistics


def rdmol_to_dataframe(mol):
    if mol is None or mol.GetNumHeavyAtoms() < 1:
        return pd.DataFrame({
            'x': [],
            'y': [],
            'z': [],
            'type': []
        })

    conf = mol.GetConformer()
    positions = np.array([np.array(conf.GetAtomPosition(i)) for
                          i in range(mol.GetNumHeavyAtoms())])
    atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for
                  i in range(mol.GetNumHeavyAtoms())]

    if len(atom_types) == 1:
        positions = positions.reshape((1, 3))

    df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'type': atom_types
    })
    if isinstance(mol, Chem.RWMol):
        df['type'] = df['type'].map({8: 0, 7: 1, 6: 2})
    return df


def the_full_monty(
        lig_mol, max_pharmacophores, area_coef,
        distance_threshold, poisson_mean, num_opportunities):
    pharm_mol = create_pharmacophore_mol(lig_mol, max_pharmacophores, area_coef)
    filtered_by_pharm_lig_dist = pharm_ligand_distance_filter(
        lig_mol, pharm_mol, threshold=2)
    filtered_by_pharm_pharm_dist = pharm_pharm_distance_filter(
        filtered_by_pharm_lig_dist)
    pharmacophore = sample_from_pharmacophores(
        filtered_by_pharm_pharm_dist, lig_mol, poisson_mean=poisson_mean,
        num_opportunities=num_opportunities)
    positive_coords = assign_mol_label(
        lig_mol, pharmacophore, threshold=distance_threshold)
    ligand_df = rdmol_to_dataframe(lig_mol)
    pharmacophore_df = rdmol_to_dataframe(pharmacophore)
    return lig_mol, pharmacophore, ligand_df, pharmacophore_df, positive_coords


def save_dfs_and_get_labels(results, output_dir):
    labels = {}
    atom_labels = defaultdict(list)
    lig_df_output_dir = mkdir(output_dir, 'parquets', 'ligands')
    pharm_df_output_dir = mkdir(output_dir, 'parquets', 'pharmacophores')
    lig_sdf_output_dir = mkdir(output_dir, 'sdf', 'ligands')
    pharm_sdf_output_dir = mkdir(output_dir, 'sdf', 'pharmacophores')

    for i in range(len(results)):
        pharm_df_fname = pharm_df_output_dir / 'pharm{}.parquet'.format(i)
        lig_df_fname = lig_df_output_dir / 'lig{}.parquet'.format(i)

        pharm_writer = Chem.SDWriter(str(Path(
            pharm_sdf_output_dir, 'pharm{}.sdf'.format(i))))
        lig_writer = Chem.SDWriter(str(Path(
            lig_sdf_output_dir, 'lig{}.sdf'.format(i))))

        lig_writer.write(results[i][0])
        pharm_writer.write(results[i][1])
        results[i][2].to_parquet(lig_df_fname)
        results[i][3].to_parquet(pharm_df_fname)

        positive_positions = results[i][-1]
        labels[i] = int(len(positive_positions) > 0)
        atom_labels[i] += positive_positions

    return labels, atom_labels


def mp_full_monty(lig_mols, output_dir,
                  max_pharmacophores, area_coef,
                  poisson_mean, num_opportunities, distance_thresholds):
    n = len(lig_mols)
    if not isinstance(max_pharmacophores, (list, tuple)):
        max_pharmacophores = [max_pharmacophores] * n
    if not isinstance(poisson_mean, (list, tuple)):
        poisson_mean = [poisson_mean] * n
    if not isinstance(distance_thresholds, (list, tuple)):
        distance_thresholds = [distance_thresholds] * n
    if not isinstance(num_opportunities, (list, tuple)):
        num_opportunities = [num_opportunities] * n
    if not isinstance(area_coef, (list, tuple)):
        area_coef = [area_coef] * n

    results = Pool().map(
        the_full_monty, lig_mols, max_pharmacophores, area_coef,
        distance_thresholds, poisson_mean, num_opportunities)
    return results


def main(args):
    sdf_loc = expand_path(args.ligands)
    output_dir = mkdir(args.output_dir)

    # can also use a directory full of individual SDF files
    print('Loading input mols')
    if sdf_loc.is_dir():
        sdfs = sdf_loc.glob('*.sdf')
        sups = [Chem.SDMolSupplier(str(sdf)) for sdf in sdfs]
        mols = [mol for sup in sups for mol in sup]
    else:
        mols = [m for m in Chem.SDMolSupplier(str(sdf_loc))]

    print('Generating pharmacophores')
    if args.use_multiprocessing:
        print('Using multiprocessing with {} cpus'.format(mp.cpu_count()))
        with Timer() as t:
            results = mp_full_monty(mols,
                                    output_dir,
                                    args.max_pharmacophores,
                                    args.area_coef,
                                    args.mean_pharmacophores,
                                    args.num_opportunities,
                                    args.distance_threshold)
        cpus = mp.cpu_count()
    else:
        with Timer() as t:
            results = [
                the_full_monty(
                    mol, args.max_pharmacophores, args.area_coef,
                    args.distance_threshold, args.mean_pharmacophores,
                    args.num_opportunities)
                for mol in mols]
        cpus = 1
    labels, atom_labels = save_dfs_and_get_labels(results, output_dir)
    save_yaml(labels, output_dir / 'labels.yaml')
    print('Fraction of positive examples: {:.3f}'.format(
        sum(labels.values()) / len(labels)))
    print('Runtime for generating {0} fake receptors: {1}'.format(
        len(mols), format_time(t.interval)))
    save_yaml(atom_labels, output_dir / 'atomic_labels.yaml')

    lig_mols = [result[0] for result in results]
    pharm_mols = [result[1] for result in results]
    with Timer() as t:
        stats = write_statistics(
            output_dir / 'stats.txt', lig_mols, pharm_mols, labels, cpus=cpus,
            args_dict=args)
    print('Runtime for gathering statistics:', format_time(t.interval))
    print()
    print(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands', type=str, help='Location of ligand sdf(s)')
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to store outputs')
    parser.add_argument('--max_pharmacophores', '-m', type=int, default=None,
                        help='Maximum number of pharmacophores for each ligand')
    parser.add_argument('--area_coef', '-a', type=float, default=None)
    parser.add_argument('--mean_pharmacophores', '-p', type=int, default=None,
                        help='Mean number of pharmacophores for each ligand')
    parser.add_argument('--num_opportunities', '-n', type=int, default=None,
                        help='Number of interaction opportunities per ligand.')
    parser.add_argument('--distance_threshold', '-t', type=float, default=3.5,
                        help='Maximum distance between ligand functional '
                             'groups and their respective pharmacophores for '
                             'the combination of the two to be considered an '
                             'active')
    parser.add_argument('--use_multiprocessing', '-mp', action='store_true',
                        help='Use multiple CPU processes')

    arguments = parser.parse_args()
    assert (bool(arguments.num_opportunities) + bool(
        arguments.mean_pharmacophores)) == 1, (
        'please specifiy precisely one of mean_pharmacophores and '
        'num_opportunities')
    assert (bool(arguments.area_coef) + bool(
        arguments.max_pharmacophores)) == 1, (
        'please specifiy precisely one of area_coef and max_opportunities')

    print()
    print('#' * os.get_terminal_size().columns)
    print(pretify_dict(vars(arguments), padding=4))
    print('#' * os.get_terminal_size().columns)
    print()
    main(arguments)
