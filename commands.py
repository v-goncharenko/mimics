import fire
import numpy as np

from mimics.classifiers import clfs_full, scores
from mimics.experiments import CrossvalidatedCsp, GridSearch
from mimics.types import Device
from mimics.utils import data_dir, default_device


def brows_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    GridSearch(
        'brows_alpha',
        data_dir / 'alpha',
        'FaExtractor',
        'brows',
        (0.2, 3.0),
        'B',
        'hypomimia',
        clfs=clfs_full,
        scores=scores,
        cv=cv,
        n_jobs=n_jobs,
        device=device,
        verbose=verbose,
    ).evaluate()


def smile_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    GridSearch(
        'smile_alpha',
        data_dir / 'alpha',
        'FaExtractor',
        'lips',
        (0.2, 3.0),
        'S',
        'hypomimia',
        clfs=clfs_full,
        scores=scores,
        cv=cv,
        n_jobs=n_jobs,
        device=device,
        verbose=verbose,
    ).evaluate()


def low_brows_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    for low in np.arange(0.1, 1, 0.1):
        GridSearch(
            'low_brows_alpha',
            data_dir / 'alpha',
            'FaExtractor',
            'brows',
            (low, 3.0),
            'B',
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def low_smile_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    for low in np.arange(0.1, 1, 0.1):
        GridSearch(
            'low_smile_alpha',
            data_dir / 'alpha',
            'FaExtractor',
            'lips',
            (low, 3.0),
            'S',
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def high_brows_alpha(
    low: float = 0.45,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = default_device,
    verbose: bool = False,
):
    for high in np.arange(2, 10, 1):
        GridSearch(
            'high_brows_alpha',
            data_dir / 'alpha',
            'FaExtractor',
            'brows',
            (low, high),
            'B',
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def high_smile_alpha(
    low: float = 0.65,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = default_device,
    verbose: bool = False,
):
    for high in np.arange(2, 10, 1):
        GridSearch(
            'high_smile_alpha',
            data_dir / 'alpha',
            'FaExtractor',
            'lips',
            (low, high),
            'S',
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def csp_brows_alpha(cv: int = 5, verbose: bool = False):
    CrossvalidatedCsp(
        'csp_brows_alpha',
        data_dir / 'alpha',
        'FaExtractor',
        'brows',
        (0.45, 5.0),
        'B',
        'hypomimia',
        cv=cv,
        verbose=verbose,
    ).evaluate()


if __name__ == "__main__":
    fire.Fire()
