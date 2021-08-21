import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import repeat
from matplotlib import pyplot as plt
from point_vs.attribution.attribution import load_model
from point_vs.attribution.attribution_fns import cam
from point_vs.utils import expand_path, load_yaml, to_numpy, mkdir
from sklearn.metrics import average_precision_score
from torch.nn.functional import one_hot

from ds_generation.position_lookup import PositionLookup


def get_stats_from_dir(model_fname, directory, no_receptor=False):
    model, model_kwargs, cmd_line_args = load_model(model_fname)
    model = model.eval()
    directory = expand_path(directory)
    atom_labels_dict = load_yaml(directory.parent / 'atomic_labels.yaml')
    mol_label_dict = load_yaml(directory.parent / 'labels.yaml')
    lig_fnames, pharm_fnames, fname_indices = [], [], []
    lig_random_precisions = []
    rec_random_precisions = []
    lig_average_precisions = []
    rec_average_precisions = []
    rec_positions = []
    lig_positions = []
    for lig_fname in directory.glob('ligands/*.parquet'):
        fname_idx = int(Path(lig_fname.name).stem.split('lig')[-1])
        if not mol_label_dict[fname_idx]:
            continue

        rec_fname = directory / 'pharmacophores' / lig_fname.name.replace(
            'lig', 'pharm')
        pharm_fnames.append(rec_fname)
        lig_fnames.append(lig_fname)
        fname_indices.append(fname_idx)
        if len(lig_fnames) == 32:
            dfs, max_ligand_val = score_batch(
                model, lig_fnames, pharm_fnames, cmd_line_args, no_receptor)
            for idx in range(len(dfs)):
                df = label_df(dfs[idx], PositionLookup(
                    atom_labels_dict[fname_indices[idx]]))
                df.sort_values(by=['y_pred'], inplace=True, ascending=False)
                lig_df = df[df['type'] < max_ligand_val]
                rec_df = df[df['type'] >= max_ligand_val]

                lig_positions += list(np.where(lig_df['y_true'] > 0.5)[0])[:1]
                rec_positions += list(np.where(rec_df['y_true'] > 0.5)[0])[:1]

                lig_random_pr = sum(lig_df['y_true']) / len(lig_df)
                rec_random_pr = sum(rec_df['y_true']) / len(rec_df)
                lig_pr = average_precision_score(
                    lig_df['y_true'], lig_df['y_pred'])
                rec_pr = average_precision_score(
                    rec_df['y_true'], rec_df['y_pred'])
                lig_random_precisions.append(lig_random_pr)
                rec_random_precisions.append(rec_random_pr)
                lig_average_precisions.append(lig_pr)
                rec_average_precisions.append(rec_pr)
            lig_fnames, pharm_fnames, fname_indices = [], [], []
    return (lig_random_precisions, lig_average_precisions,
            rec_random_precisions, rec_average_precisions,
            lig_positions, rec_positions)


def get_collate_fn(dim):
    """Processing of inputs which takes place after batch is selected.

    LieConv networks take tuples of torch tensors (p, v, m), which are:
        p, (batch_size, n_atoms, 3): coordinates of each atom
        v, (batch_size, n_atoms, n_features): features for each atom
        m, (batch_size, n_atoms): mask for each coordinate slot

    Note that n_atoms is the largest number of atoms in a structure in
    each batch.

    Arguments:
        dim: size of input (node) features

    Returns:
        Pytorch collate function
    """

    def collate(batch, max_len):
        batch_size = len(batch)
        p_batch = torch.zeros(batch_size, max_len, 3)
        v_batch = torch.zeros(batch_size, max_len, dim)
        m_batch = torch.zeros(batch_size, max_len)
        for batch_index, (p, v) in enumerate(batch):
            size = p.shape[1]
            p_batch[batch_index, :size, :] = p
            v_batch[batch_index, :size, :] = v
            m_batch[batch_index, :size] = 1
        return p_batch.cuda(), v_batch.cuda(), m_batch.bool().cuda()

    return collate


def get_distances(lst):
    res = []
    for i in range(len(lst) // 2):
        res.append(
            np.linalg.norm(np.array(lst[2 * i]) - np.array(lst[2 * i + 1])))


def score_batch(model, lig_fnames, pharm_fnames, cmd_line_args, no_receptor):
    collate_fn = get_collate_fn(13)
    batch = []
    dfs = []
    max_len = 0
    max_ligand_val = np.inf
    for pharm_fname, lig_fname in zip(pharm_fnames, lig_fnames):
        df, p, v, _, max_ligand_val = parse_parquets(
            lig_fname, pharm_fname, cmd_line_args, no_receptor)
        batch.append((p, v))
        max_len = max(max_len, len(df))
        dfs.append(df)
    batch = collate_fn(batch, max_len)
    attributions = cam(model, *batch)
    res = []
    for idx, df in enumerate(dfs):
        df['y_pred'] = attributions[idx, :len(df), :].squeeze()
        res.append(df)
    return dfs, max_ligand_val


def score_atoms(
        model, lig_fname, pharm_fname, cmd_line_args, no_receptor=False):
    df, p, v, m, max_ligand_val = parse_parquets(
        lig_fname, pharm_fname, cmd_line_args, no_receptor)
    model_labels = cam(model, p.cuda(), v.cuda(), m.cuda(), bs=1)
    df['y_pred'] = model_labels
    return df, float(to_numpy(
        torch.sigmoid(model((p.cuda(), v.cuda(), m.cuda()))[0, ...])))


def label_df(df, positions_list):
    labels = []
    coords_np = np.vstack([
        df['x'].to_numpy(),
        df['y'].to_numpy(),
        df['z'].to_numpy()
    ]).T
    for i in range(len(df)):
        labels.append(int(list(coords_np[i, :]) in positions_list))
    df['y_true'] = labels
    return df


def parse_parquets(lig_fname, pharm_fname, cmd_line_args, no_receptor=False):
    polar_hydrogens = cmd_line_args['hydrogens']
    # C N O F P S Cl, Br, I
    recognised_atomic_numbers = (6, 7, 8, 9, 15, 16, 17, 35, 53)
    atomic_number_to_index = defaultdict(
        lambda: max_ligand_feature_id)
    atomic_number_to_index.update({
        num: idx for idx, num in enumerate(recognised_atomic_numbers)
    })

    if polar_hydrogens:
        atomic_number_to_index.update({
            1: max(atomic_number_to_index.values()) + 1
        })

    # +1 to accommodate for unmapped elements
    max_ligand_feature_id = max(atomic_number_to_index.values()) + 1

    # Any other elements not accounted for given a category of their own
    feature_dim = (max_ligand_feature_id + 1) + 3

    lig_struct = pd.read_parquet(lig_fname)
    if not polar_hydrogens:
        lig_struct = lig_struct[lig_struct['type'] > 1]
    lig_struct.type = lig_struct['type'].map(
        atomic_number_to_index)

    if no_receptor:
        df = lig_struct
    else:
        pharm_struct = pd.read_parquet(pharm_fname)
        pharm_struct['type'] += max_ligand_feature_id
        df = pd.concat([pharm_struct, lig_struct])

    p = torch.from_numpy(
        np.expand_dims(df[df.columns[:3]].to_numpy(), 0)).float()

    v = one_hot(torch.from_numpy(np.expand_dims(df.type.to_numpy(), 0)).long(),
                feature_dim)
    m = repeat(torch.from_numpy(np.ones((len(df),))).bool(), 'n -> b n', b=1)

    return df, p.float(), v.float(), m, max_ligand_feature_id


def plot_rank_histogram(lig_ranks, rec_ranks, fname=None):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    max_rank = max(lig_ranks + rec_ranks)
    for idx, (ranks, title) in enumerate(zip(
            [lig_ranks, rec_ranks], ['Ligand', 'Receptor'])):
        axs[idx].hist(ranks, density=False, bins=list(range(max_rank + 1)),
                      edgecolor='black', linewidth=1.0, color='blue')
        axs[idx].set_title(title)
        axs[idx].set_xlabel('Top-scoring bonding atom rank')
        axs[idx].set_ylabel('Count')
    fig.tight_layout()
    if fname is not None:
        fname = expand_path(fname)
        mkdir(fname.parent)
        fig.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Saved pytorch model weights')
    parser.add_argument('input_dir', type=str,
                        help='Location of ligand and receptor parquet files')
    parser.add_argument('--no_receptor', '-n', action='store_true',
                        help='Do not include receptor information')
    args = parser.parse_args()

    lrp, lap, rrp, rap, lig_positions, rec_positions = get_stats_from_dir(
        args.model, args.input_dir, args.no_receptor)
    project_name = Path(args.model).parents[1].name
    print()
    print('Project:', project_name)
    print('Mean average precision (ligand):               {:.4f}'.format(
        np.mean(lap)))
    print('Random average precision (ligand):             {:.4f}'.format(
        np.mean(lrp)))
    print('Mean average precision (receptor):             {:.4f}'.format(
        np.mean(rap)))
    print('Random average precision (receptor):           {:.4f}'.format(
        np.mean(rrp)))
    print()
    print('Mean top scoring bonding atom rank (ligand):   {:.4f}'.format(
        np.mean(lig_positions)))
    print('Mean top scoring bonding atom rank (receptor): {:.4f}'.format(
        np.mean(rec_positions)))
    plot_rank_histogram(lig_positions, rec_positions,
                        'rank_histogram_{}.png'.format(project_name))
