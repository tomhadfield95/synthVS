#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic pharmacophores. Code written by, or modified from code
written by, Tom Hadfield.
"""
import argparse

from point_vs.utils import expand_path, mkdir, save_yaml
from rdkit import Chem

from ds_generation.filters import sample_from_pharmacophores, \
    pharm_pharm_distance_filter, pharm_ligand_distance_filter
from ds_generation.generate import create_pharmacophore_mol
from ds_generation.label import assign_label


def save_list_of_mols(mols, loc):
    w = Chem.SDWriter(loc)
    for m in mols:
        w.write(m)

    return 0


def main(args):
    sdf_loc = expand_path(args.ligands)
    output_dir = mkdir(args.output_dir)
    lig_output_dir = mkdir(output_dir / 'ligands')
    rec_output_dir = mkdir(output_dir / 'receptors')

    # can also use a directory full of individual SDF files
    print('Loading input mols')
    if sdf_loc.is_dir():
        sdfs = sdf_loc.glob('*.sdf')
        sups = [Chem.SDMolSupplier(sdf) for sdf in sdfs]
        mols = [mol for sup in sups for mol in sup]
    else:
        mols = [m for m in Chem.SDMolSupplier(sdf_loc)]

    print('Generating initial pharmacophores')
    pharm_mols = [
        create_pharmacophore_mol(m, args.init_sample_size) for m in mols]

    print('Applying filters')
    filtered_by_pharm_lig_dist = [
        pharm_ligand_distance_filter(mols[i], pharm_mols[i])
        for i in range(len(pharm_mols))]

    filtered_by_pharm_pharm_dist = [
        pharm_pharm_distance_filter(filtered_by_pharm_lig_dist[i])
        for i in range(len(filtered_by_pharm_lig_dist))]

    randomly_sampled_subset = [
        sample_from_pharmacophores(m, args.max_pharmacophores)
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
    parser.add_argument('--init_sample_size', '-i', type=int, default=20,
                        help='Initial number of pharmacophores to sample '
                             'before filtering')
    parser.add_argument('--max_pharmacophores', '-m', type=int, default=6,
                        help='Maximum number of pharmacophores for each ligand')
    parser.add_argument('--distance_threshold', '-t', type=float, default=3.5,
                        help='Minimum distance between ligand functional '
                             'groups and their respective pharmacophores for '
                             'the combination of the two to be considered an '
                             'active')

    parser.add_argument()
    arguments = parser.parse_args()
    main(arguments)
