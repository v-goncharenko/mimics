import fire
import numpy as np

from mimics.classifiers import clfs_full, scores
from mimics.experiments import (
    CrossvalidatedCsp,
    DrawKeypoints,
    GridSearch,
    WindowedCorrelations,
)
from mimics.transformers.extractors import inds_68
from mimics.types import Device
from mimics.utils import data_dir


points = {'brows': 'brows', 'smile': 'lips'}
exercises = {'brows': 'B', 'smile': 'S'}
cutoffs = {'brows': (0.45, 5.0), 'smile': (0.65, 6.0)}


def gridsearch(
    dataset: str,
    exercise: str,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = 'cpu',
    verbose: bool = False,
):
    GridSearch(
        f'{exercise}_{dataset}',
        data_dir / dataset,
        'FaExtractor',
        points[exercise],
        cutoffs[exercise],
        exercises[exercise],
        'hypomimia',
        clfs=clfs_full,
        scores=scores,
        cv=cv,
        n_jobs=n_jobs,
        device=device,
        verbose=verbose,
    ).evaluate()


def low_gridsearch(
    dataset: str,
    exercise: str,
    high: float = 4.0,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = 'cpu',
    verbose: bool = False,
):
    for low in np.arange(0.1, 1, 0.1):
        GridSearch(
            f'low_{exercise}_{dataset}',
            data_dir / dataset,
            'FaExtractor',
            points[exercise],
            (low, high),
            exercises[exercise],
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def high_gridsearch(
    dataset: str,
    exercise: str,
    low: float = 0.4,
    cv: int = 5,
    n_jobs: int = 1,
    device: Device = 'cpu',
    verbose: bool = False,
):
    for high in np.arange(2, 10, 1):
        GridSearch(
            f'high_{exercise}_{dataset}',
            data_dir / dataset,
            'FaExtractor',
            points[exercise],
            (low, high),
            exercises[exercise],
            'hypomimia',
            clfs=clfs_full,
            scores=scores,
            cv=cv,
            n_jobs=n_jobs,
            device=device,
            verbose=verbose,
        ).evaluate()


def plot_csp(dataset: str, exercise: str, cv: int = 5, verbose: bool = False):
    CrossvalidatedCsp(
        f'csp_{exercise}_{dataset}',
        data_dir / dataset,
        'FaExtractor',
        points[exercise],
        (0.45, 3.0),
        exercises[exercise],
        'hypomimia',
        cv=cv,
        verbose=verbose,
        log=False,
    ).evaluate()


def corrs_brows(dataset: str, verbose: bool = False):
    WindowedCorrelations(
        f'corrs_brows_{dataset}',
        data_dir / dataset,
        'FaExtractor',
        'all',
        (0.45, 5.0),
        exercises['brows'],
        'hypomimia',
        windows=(1, 1.5, 2),
        axes=(1,),
        right_points=inds_68['right_brow'],
        left_points=inds_68['left_brow'],
        verbose=verbose,
        log=False,
    ).evaluate()


def corrs_smile(dataset: str, verbose: bool = False):
    WindowedCorrelations(
        f'corrs_smile_{dataset}',
        data_dir / dataset,
        'FaExtractor',
        'all',
        (0.65, 6.0),
        exercises['smile'],
        'hypomimia',
        windows=(1.5,),
        axes=(0, 1),
        right_points=inds_68['right_lips'],
        left_points=inds_68['left_lips'],
        verbose=verbose,
        log=False,
    ).evaluate()


def draw_points(dataset: str, n_jobs: int = 1, verbose: bool = False):
    DrawKeypoints(
        data_dir / dataset, cutoff=2.5, n_jobs=n_jobs, verbose=verbose
    ).evaluate()


if __name__ == "__main__":
    fire.Fire()
