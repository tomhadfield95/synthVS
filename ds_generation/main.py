#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic pharmacophores. Code written by, or modified from code
written by, Tom Hadfield.
"""
import argparse

from pathos.multiprocessing import ProcessingPool as Pool
from point_vs.utils import expand_path, mkdir, save_yaml
from rdkit import Chem

from filters import sample_from_pharmacophores, \
    pharm_pharm_distance_filter, pharm_ligand_distance_filter
from generate import create_pharmacophore_mol
from label import assign_label


def save_list_of_mols(mols, loc):
    w = Chem.SDWriter(loc)
    for m in mols:
        w.write(m)

    return 0


def the_full_monty(
        lig_mol, max_pharmacophores, mean_pharmacophores, distance_threshold):
    pharm_mol = create_pharmacophore_mol(lig_mol, max_pharmacophores)
    filtered_by_pharm_lig_dist = pharm_ligand_distance_filter(
        lig_mol, pharm_mol, threshold=2)
    filtered_by_pharm_pharm_dist = pharm_pharm_distance_filter(
        filtered_by_pharm_lig_dist)
    randomly_sampled_subset = sample_from_pharmacophores(
        filtered_by_pharm_pharm_dist, mean_pharmacophores)
    ligand, pharmacophore, label = assign_label(
        lig_mol, randomly_sampled_subset, threshold=distance_threshold)
    return ligand, pharmacophore, label


def mp_full_monty(lig_mols, lig_output_dir, pharm_output_dir,
                  max_pharmacophores, mean_pharmacophores, distance_thresholds):
    lig_output_dir = mkdir(lig_output_dir)
    pharm_output_dir = mkdir(pharm_output_dir)
    n = len(lig_mols)
    if not isinstance(max_pharmacophores, (list, tuple)):
        max_pharmacophores = [max_pharmacophores] * n
    if not isinstance(mean_pharmacophores, (list, tuple)):
        mean_pharmacophores = [mean_pharmacophores] * n
    if not isinstance(distance_thresholds, (list, tuple)):
        distance_thresholds = [distance_thresholds] * n

    results = Pool().map(
        the_full_monty, lig_mols, max_pharmacophores,
        mean_pharmacophores, distance_thresholds)

    labels = {}
    for i in range(len(results)):
        lig_writer = Chem.SDWriter(str(lig_output_dir / 'lig{}.sdf'.format(i)))
        pharm_writer = Chem.SDWriter(
            str(pharm_output_dir / 'pharm{}.sdf'.format(i)))
        lig_writer.write(results[i][0])
        pharm_writer.write(results[i][1])
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

    if args.use_multiprocessing:
        mp_full_monty(mols,
                      output_dir / 'ligands',
                      output_dir / 'pharmacophores',
                      args.max_pharmacophores,
                      args.mean_pharmacophores,
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
            sample_from_pharmacophores(m, args.mean_pharmacophores)
            for m in filtered_by_pharm_pharm_dist]

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
    parser.add_argument('--mean_pharmacophores', '-p', type=int, default=6,
                        help='Mean number of pharmacophores for each ligand')
    parser.add_argument('--distance_threshold', '-t', type=float, default=3.5,
                        help='Maximum distance between ligand functional '
                             'groups and their respective pharmacophores for '
                             'the combination of the two to be considered an '
                             'active')
    parser.add_argument('--use_multiprocessing', '-mp', action='store_true',
                        help='Use multiple CPU processes')

    arguments = parser.parse_args()
    main(arguments)
