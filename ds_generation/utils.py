"""
Some basic helper functions for formatting time and sticking dataframes
together.
"""
import copy
import math
import multiprocessing as mp
import shutil
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


def pretify_dict(d, padding=5):
    max_key_len = max([len(str(key)) for key in d.keys()])
    line_len = max_key_len + padding
    s = ''
    for key, value in d.items():
        spaces = ' ' * (line_len - len(str(key)))
        s += '{0}:{1}{2}\n'.format(
            key, spaces, value
        )
    return s[:-1]


def save_yaml(d, fname):
    """Save a dictionary in yaml format."""
    with open(Path(fname).expanduser(), 'w', encoding='utf-8') as f:
        yaml.dump(d, stream=f)


def load_yaml(fname):
    """Load a yaml dictionary"""
    # For backwards compatability reasons we should ignore missing constructors
    yaml.add_multi_constructor(
        '',
        lambda loader, suffix, node: None, Loader=yaml.SafeLoader)
    with open(Path(fname).expanduser(), 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def mkdir(*paths):
    """Make a new directory, including parents."""
    path = Path(*paths).expanduser().resolve()
    path.mkdir(exist_ok=True, parents=True)
    return path


def expand_path(*paths):
    return Path(*paths).expanduser().resolve()


def format_time(t):
    """Returns string continaing time in hh:mm:ss format.

    Arguments:
        t: time in seconds

    Raises:
        ValueError if t < 0
    """
    t = t or 0
    if t < 0:
        raise ValueError('Time must be positive.')

    t = int(math.floor(t))
    h = t // 3600
    m = (t - (h * 3600)) // 60
    s = t - ((h * 3600) + (m * 60))
    return '{0:02d}:{1:02d}:{2:02d}'.format(h, m, s)

class Timer:
    """Simple timer class.

    To time a block of code, wrap it like so:

        with Timer() as t:
            <some_code>
        total_time = t.interval

    The time taken for the code to execute is stored in t.interval.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
