#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic pharmacophores. Code written by, or modified from code
written by, Tom Hadfield.
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from point_vs.utils import expand_path, mkdir, save_yaml, pretify_dict
from rdkit import Chem

from filters import sample_from_pharmacophores, \
    pharm_pharm_distance_filter, pharm_ligand_distance_filter
from generate import create_pharmacophore_mol
from label import assign_label


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
        lig_mol, max_pharmacophores, distance_threshold, poisson_mean,
        num_opportunities):
    pharm_mol = create_pharmacophore_mol(lig_mol, max_pharmacophores)
    filtered_by_pharm_lig_dist = pharm_ligand_distance_filter(
        lig_mol, pharm_mol, threshold=2)
    filtered_by_pharm_pharm_dist = pharm_pharm_distance_filter(
        filtered_by_pharm_lig_dist)
    randomly_sampled_subset = sample_from_pharmacophores(
        filtered_by_pharm_pharm_dist, lig_mol, poisson_mean=poisson_mean,
        num_opportunities=num_opportunities)
    ligand, pharmacophore, label = assign_label(
        lig_mol, randomly_sampled_subset, threshold=distance_threshold)
    ligand_df = rdmol_to_dataframe(ligand)
    pharmacophore_df = rdmol_to_dataframe(pharmacophore)
    return ligand_df, pharmacophore_df, label


def mp_full_monty(lig_mols, lig_output_dir, pharm_output_dir,
                  max_pharmacophores, poisson_mean, num_opportunities,
                  distance_thresholds):
    lig_output_dir = mkdir(lig_output_dir)
    pharm_output_dir = mkdir(pharm_output_dir)
    n = len(lig_mols)
    if not isinstance(max_pharmacophores, (list, tuple)):
        max_pharmacophores = [max_pharmacophores] * n
    if not isinstance(poisson_mean, (list, tuple)):
        poisson_mean = [poisson_mean] * n
    if not isinstance(distance_thresholds, (list, tuple)):
        distance_thresholds = [distance_thresholds] * n
    if not isinstance(num_opportunities, (list, tuple)):
        num_opportunities = [num_opportunities] * n

    results = Pool().map(
        the_full_monty, lig_mols, max_pharmacophores, distance_thresholds,
        poisson_mean, num_opportunities)

    labels = {}
    for i in range(len(results)):
        pharm_fname = pharm_output_dir / 'pharm{}.parquet'.format(i)
        lig_fname = lig_output_dir / 'lig{}.parquet'.format(i)
        results[i][0].to_parquet(lig_fname)
        results[i][1].to_parquet(pharm_fname)
        labels[i] = results[i][2]
    save_yaml(labels, lig_output_dir.parent / 'labels.yaml')


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
        print(len(mols))
        mp_full_monty(mols,
                      output_dir / 'ligands',
                      output_dir / 'pharmacophores',
                      args.max_pharmacophores,
                      args.mean_pharmacophores,
                      args.num_opportunities,
                      args.distance_threshold)
    else:
        lig_output_dir = mkdir(output_dir / 'ligands')
        rec_output_dir = mkdir(output_dir / 'pharmacophores')
        print('Generating initial pharmacophores')
        pharm_mols = [
            create_pharmacophore_mol(m, args.max_pharmacophores) for m in mols]

        print('Applying filters')
        filtered_by_pharm_lig_dist = [
            pharm_ligand_distance_filter(mols[i], pharm_mols[i])
            for i in range(len(pharm_mols))]

        filtered_by_pharm_pharm_dist = [
            pharm_pharm_distance_filter(filtered_by_pharm_lig_dist[i])
            for i in range(len(filtered_by_pharm_lig_dist))]

        randomly_sampled_subset = [
            sample_from_pharmacophores(
                filtered_by_pharm_pharm_dist[i], mols[i],
                args.mean_pharmacophores, args.num_opportunities)
            for i in range(len(filtered_by_pharm_pharm_dist))]

        labels = {}
        for i in range(len(randomly_sampled_subset)):
            lig, pharm, label = assign_label(
                mols[i], randomly_sampled_subset[i],
                threshold=args.distance_threshold)
            lig_writer = Chem.SDWriter(
                str(lig_output_dir / 'lig{}.sdf'.format(i)))
            pharm_writer = Chem.SDWriter(
                str(rec_output_dir / 'pharm{}.sdf'.format(i)))
            lig_writer.write(lig)
            pharm_writer.write(pharm)
            labels[i] = label
        save_yaml(labels, output_dir / 'labels.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ligands', type=str, help='Location of ligand sdf(s)')
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to store outputs')
    parser.add_argument('--max_pharmacophores', '-m', type=int, default=20,
                        help='Maximum number of pharmacophores for each ligand')
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

    print()
    print('#' * os.get_terminal_size().columns)
    print(pretify_dict(vars(arguments), padding=4))
    print('#' * os.get_terminal_size().columns)
    print()
    main(arguments)
