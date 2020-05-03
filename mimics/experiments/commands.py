from pathlib import Path

import fire

from ..classifiers import clfs, scores
from .experiment import Experiment


datasets_dir = Path(__file__).resolve().parent.parent.parent / 'data'


def brows_alpha(cv: int = 5, n_jobs: int = 1, verbose: bool = False):
    Experiment(
        'brows_alpha',
        datasets_dir / 'alpha',
        'FaExtractor',
        'brows',
        'B',
        'hypomimia',
        clfs,
        scores,
        cv,
        n_jobs=n_jobs,
        verbose=True,
    ).evaluate()


def smile_alpha(cv: int = 5, n_jobs: int = 1, verbose: bool = False):
    Experiment(
        'smile_alpha',
        datasets_dir / 'alpha',
        'FaExtractor',
        'lips',
        'S',
        'hypomimia',
        clfs,
        scores,
        5,
        n_jobs=n_jobs,
        verbose=True,
    ).evaluate()


if __name__ == "__main__":
    fire.Fire()
