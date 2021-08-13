"""
DataLoaders to take parquet directories and create feature vectors suitable
for use by models found in this project.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from point_vs.preprocessing.data_loaders import get_collate_fn
from point_vs.preprocessing.preprocessing import uniform_random_rotation
from point_vs.utils import load_yaml
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_data_loader(
        *data_roots, receptors=None, batch_size=32, rot=True,
        polar_hydrogens=True, mode='train', no_receptor=False):
    """Give a DataLoader from a list of receptors and data roots."""
    ds_kwargs = {
        'rot': rot
    }
    ds = multiple_source_dataset(
        *data_roots, balanced=True, polar_hydrogens=polar_hydrogens,
        receptors=receptors, no_receptor=no_receptor, **ds_kwargs)
    collate = get_collate_fn(ds.feature_dim)
    sampler = ds.sampler if mode == 'train' else None
    return DataLoader(
        ds, batch_size, False, sampler=sampler,  # num_workers=mp.cpu_count(),
        collate_fn=collate, drop_last=False, pin_memory=True)


def multiple_source_dataset(*base_paths, receptors=None, polar_hydrogens=True,
                            balanced=True, no_receptor=False, **kwargs):
    """Concatenate mulitple datasets into one, preserving balanced sampling.

    Arguments:
        base_paths: locations of parquets files, one for each dataset.
        receptors: receptors to include. If None, all receptors found are used.
        polar_hydrogens:
        balanced:

    Returns:
        Concatenated dataset including balanced sampler.
    """
    datasets = []
    labels = []
    filenames = []
    base_paths = sorted(
        [Path(bp).expanduser() for bp in base_paths if bp is not None])
    for base_path in base_paths:
        if base_path is not None:
            dataset = SynthPharmDataset(
                base_path, receptors=receptors, polar_hydrogens=polar_hydrogens,
                no_receptor=no_receptor, **kwargs)
            labels += list(dataset.labels)
            filenames += dataset.filenames
            datasets.append(dataset)
    labels = np.array(labels)
    active_count = np.sum(labels)
    class_sample_count = np.array(
        [len(labels) - active_count, active_count])
    if np.sum(labels) == len(labels) or np.sum(labels) == 0 or not balanced:
        sampler = None
    else:
        weights = 1. / class_sample_count
        sample_weights = torch.from_numpy(
            np.array([weights[i] for i in labels]))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    multi_source_dataset = torch.utils.data.ConcatDataset(datasets)
    multi_source_dataset.sampler = sampler
    multi_source_dataset.class_sample_count = class_sample_count
    multi_source_dataset.labels = labels
    multi_source_dataset.filenames = filenames
    multi_source_dataset.base_path = ', '.join([str(bp) for bp in base_paths])
    multi_source_dataset.feature_dim = datasets[0].feature_dim
    return multi_source_dataset


class SynthPharmDataset(torch.utils.data.Dataset):
    """Class for feeding structure parquets into network."""

    def __init__(
            self, base_path, polar_hydrogens=True, rot=False, receptors=None,
            no_receptor=False, **kwargs):
        """Initialise dataset.

        Arguments:
            base_path: path containing the 'receptors' and 'ligands'
                directories, which in turn contain <rec_name>.parquets files
                and folders called <rec_name>_[active|decoy] which in turn
                contain <ligand_name>.parquets files. All parquets files from
                this directory are recursively loaded into the dataset.
            polar_hydrogens: include polar hydrogens as input
            rot: random rotation of inputs
            receptors:
            kwargs: keyword arguments passed to the parent class (Dataset).
        """

        super().__init__(**kwargs)
        self.base_path = Path(base_path).expanduser()
        self.no_receptor = no_receptor

        if not self.base_path.exists():
            raise FileNotFoundError(
                'Dataset {} does not exist.'.format(self.base_path))
        self.polar_hydrogens = polar_hydrogens

        print('Loading all structures in', self.base_path)
        filenames = list(
            (self.base_path / 'ligands').glob('*.parquet'))

        filenames = sorted(filenames)
        labels = []
        labels_dict = load_yaml(self.base_path / 'labels.yaml')
        for fname in filenames:
            idx = int(Path(fname.name).stem.split('lig')[-1])
            labels.append(labels_dict[idx])

        self.pre_aug_ds_len = len(filenames)
        self.filenames = filenames

        labels = np.array(labels)
        active_count = np.sum(labels)
        class_sample_count = np.array(
            [len(labels) - active_count, active_count])
        if np.sum(labels) == len(labels) or np.sum(labels) == 0:
            self.sampler = None
        else:
            weights = 1. / class_sample_count
            self.sample_weights = torch.from_numpy(
                np.array([weights[i] for i in labels]))
            self.sampler = torch.utils.data.WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights)
            )
        self.labels = labels
        print('There are', len(labels), 'training points in', base_path)

        # apply random rotations to ALL coordinates?
        self.transformation = uniform_random_rotation if rot else lambda x: x

        # C N O F P S Cl, Br, I
        recognised_atomic_numbers = (6, 7, 8, 9, 15, 16, 17, 35, 53)
        atomic_number_to_index = {
            num: idx for idx, num in enumerate(recognised_atomic_numbers)
        }

        if self.polar_hydrogens:
            atomic_number_to_index.update({
                1: max(atomic_number_to_index.values()) + 1
            })

        # +1 to accommodate for unmapped elements
        self.max_ligand_feature_id = max(atomic_number_to_index.values()) + 1

        # Any other elements not accounted for given a category of their own
        self.atomic_number_to_index = defaultdict(
            lambda: self.max_ligand_feature_id)
        self.atomic_number_to_index.update(atomic_number_to_index)
        self.feature_dim = (self.max_ligand_feature_id + 1) + 3

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.filenames)

    def __getitem__(self, item):
        """Given an index, locate and preprocess relevant parquet file.

        Arguments:
            item: index in the list of filenames denoting which ligand and
                receptor to fetch

        Returns:
            Tuple containing (a) a tuple with a list of tensors: cartesian
            coordinates, feature vectors and masks for each point, as well as
            the number of points in the structure and (b) the label \in \{0, 1\}
            denoting whether the structure is an active or a decoy.
        """

        lig_fname = self.filenames[item]
        label = self.labels[item]
        mol_idx = str(Path(lig_fname.name).stem).split('lig')[-1]
        pharm_fname = Path(
            self.base_path, 'pharmacophores', 'pharm{}.parquet'.format(mol_idx))
        if not pharm_fname.is_file():
            raise RuntimeError(
                'Receptor for ligand {0} not found. Looking for file '
                'named {1}'.format(lig_fname, pharm_fname))

        lig_struct = pd.read_parquet(lig_fname)
        if not self.polar_hydrogens:
            lig_struct = lig_struct[lig_struct['type'] > 1]
        lig_struct.type = lig_struct['type'].map(
            self.atomic_number_to_index)

        if self.no_receptor:
            struct = lig_struct
        else:
            pharm_struct = pd.read_parquet(pharm_fname)
            pharm_struct['type'] += self.max_ligand_feature_id
            struct = pd.concat([pharm_struct, lig_struct])

        p = torch.from_numpy(
            np.expand_dims(self.transformation(
                struct[struct.columns[:3]].to_numpy()), 0))

        v = one_hot(torch.from_numpy(struct.type.to_numpy()).long(),
                    self.feature_dim)

        return (p, v, len(struct)), lig_fname, pharm_fname, label
