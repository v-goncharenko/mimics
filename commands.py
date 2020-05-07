from pathlib import Path

import fire
import numpy as np

from mimics.classifiers import clfs, scores
from mimics.experiments import Experiment
from mimics.types import Device
from mimics.utils import default_device


datasets_dir = Path(__file__).resolve().parent / 'data'


def brows_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    Experiment(
        'brows_alpha',
        datasets_dir / 'alpha',
        'FaExtractor',
        'brows',
        (0.2, 3.0),
        'B',
        'hypomimia',
        clfs,
        scores,
        cv,
        n_jobs=n_jobs,
        device=device,
        verbose=True,
    ).evaluate()


def smile_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    Experiment(
        'smile_alpha',
        datasets_dir / 'alpha',
        'FaExtractor',
        'lips',
        (0.2, 3.0),
        'S',
        'hypomimia',
        clfs,
        scores,
        5,
        n_jobs=n_jobs,
        device=device,
        verbose=True,
    ).evaluate()


def low_brows_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    for low in np.arange(0.1, 1, 0.1):
        Experiment(
            'low_brows_alpha',
            datasets_dir / 'alpha',
            'FaExtractor',
            'brows',
            (low, 3.0),
            'B',
            'hypomimia',
            clfs,
            scores,
            cv,
            n_jobs=n_jobs,
            device=device,
            verbose=True,
        ).evaluate()


def low_smile_alpha(
    cv: int = 5, n_jobs: int = 1, device: Device = default_device, verbose: bool = False
):
    for low in np.arange(0.1, 1, 0.1):
        Experiment(
            'low_smile_alpha',
            datasets_dir / 'alpha',
            'FaExtractor',
            'lips',
            (low, 3.0),
            'S',
            'hypomimia',
            clfs,
            scores,
            cv,
            n_jobs=n_jobs,
            device=device,
            verbose=True,
        ).evaluate()


def high_brows_alpha(
    low: float = 0.45,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = default_device,
    verbose: bool = False,
):
    for high in np.arange(2, 10, 1):
        Experiment(
            'high_brows_alpha',
            datasets_dir / 'alpha',
            'FaExtractor',
            'brows',
            (low, high),
            'B',
            'hypomimia',
            clfs,
            scores,
            cv,
            n_jobs=n_jobs,
            device=device,
            verbose=True,
        ).evaluate()


def high_smile_alpha(
    low: float = 0.65,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = default_device,
    verbose: bool = False,
):
    for high in np.arange(2, 10, 1):
        Experiment(
            'high_smile_alpha',
            datasets_dir / 'alpha',
            'FaExtractor',
            'lips',
            (low, high),
            'S',
            'hypomimia',
            clfs,
            scores,
            cv,
            n_jobs=n_jobs,
            device=device,
            verbose=True,
        ).evaluate()


if __name__ == "__main__":
    fire.Fire()
