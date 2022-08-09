import argparse
import subprocess
from pathlib import Path
import os
import shutil

def mkdir(*paths):
    """Make a new directory, including parents."""
    path = Path(*paths).expanduser().resolve()
    path.mkdir(exist_ok=True, parents=True)
    return path


def copytree(src, dst, symlinks=False, ignore=None):
    if not Path(src).is_dir() and not Path(src).is_file():
        return
    for item in os.listdir(src):
        s = Path(src, item)
        d = Path(dst, item)

        if os.path.isdir(s):
            d.mkdir(parents=True, exist_ok=True)
            shutil.copytree(s, d, symlinks, ignore)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str,
            help='directory (should be named a/b/c/parquets)')
    parser.add_argument('output_dir',type=str, 
            help='output_directory')

    args = parser.parse_args()
    base_path = Path(args.base_path).expanduser().resolve()


    lig_path = base_path / 'ligands'
    rec_path = base_path / 'pharmacophores'

    train_path = mkdir(f'{args.output_dir}/train')
    test_path = mkdir(f'{args.output_dir}/test')

    train_ligs = mkdir(train_path / 'ligands')
    train_recs = mkdir(train_path / 'pharmacophores')

    test_ligs = mkdir(test_path / 'ligands')
    test_recs = mkdir(test_path / 'pharmacophores')

    for idx, lig in enumerate(lig_path.glob('*.parquet')):
        train = bool(idx % 5)
        print(idx, train)
        lig_idx = Path(lig.name).stem.split('lig')[-1]
        rec = rec_path / 'pharm{}.parquet'.format(lig_idx)
        if train:
            shutil.copy(lig, train_ligs / lig.name)
            shutil.copy(rec, train_recs / rec.name)
        else:
            shutil.copy(lig, test_ligs / lig.name)
            shutil.copy(rec, test_recs / rec.name)
