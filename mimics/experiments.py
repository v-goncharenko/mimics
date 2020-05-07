from collections import namedtuple
from dataclasses import dataclass

import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate

from .datasets import FaceLandmarksDataset
from .transformers import extractors, get_preprocessing
from .types import Device, Directory, Tuple, Union
from .utils import default_device


@dataclass
class Experiment(object):
    '''
    Args:
        name: name of the experiment for MLflow
        extractor: classname of Extractor see `.tranformers.extractors`
        points: param for `get_preprocessing`
        exercise: 'B' for brows, 'S' for smile
        labeling: see `get_labels` method
        clfs: dict of classifiers and their params e.g. `.classifiers.clfs`
        scores: tuple of scores e.g. `.classifiers.scores`
        log: enable MLflow logging
    '''

    name: str
    dataset_dir: Directory
    extractor: str
    points: Union[str, tuple]
    cutoffs: Tuple[float, float]
    exercise: str
    labeling: str
    clfs: dict
    scores: tuple
    cv: int = 5
    n_jobs: int = 1
    device: Device = default_device
    verbose: bool = False
    log: bool = True

    def evaluate(self):
        if self.log:
            mlflow.set_experiment(self.name)
        self.state = namedtuple('State', 'dataset cv mask labels features')

        extr = self.get_extractor()
        preproc = get_preprocessing(self.points, *self.cutoffs)
        self.state.dataset = FaceLandmarksDataset(self.dataset_dir, extr, preproc)
        self.state.cv = StratifiedKFold(self.cv, True)
        self.state.mask = self.state.dataset.markup['exercise'] == self.exercise
        self.state.labels = self.get_labels()
        self.state.labels = self.state.labels[self.state.mask]
        self.state.features = self.state.dataset.data[self.state.mask]

        for clf, params in self.clfs.values():
            for param_set in ParameterGrid(params):
                if self.verbose:
                    print(clf.name, param_set)
                cv_res = cross_validate(
                    clone(clf).set_params(**param_set),
                    self.state.features,
                    self.state.labels,
                    scoring=self.scores,
                    cv=self.state.cv,
                    n_jobs=self.n_jobs,
                )
                if self.log:
                    self.log_result(clf, param_set, cv_res)

    def log_result(self, clf, params, cv_res):
        with mlflow.start_run():
            mlflow.log_params(
                {
                    'dataset': self.state.dataset.name,
                    'extractor': self.extractor,
                    'points': self.points,
                    'low': min(self.cutoffs),
                    'high': max(self.cutoffs),
                    'exercise': self.exercise,
                    'labeling': self.labeling,
                    'cv': self.cv,
                    'clf': clf.name,
                    **params,
                }
            )

            for aggr in (np.mean, np.std, np.median):
                mlflow.log_metrics(
                    {
                        f'{aggr.__name__}_{score}': aggr(cv_res[f'test_{score}'])
                        or np.nan
                        for score in self.scores
                    }
                )
            mlflow.log_metrics(
                {
                    'records': len(self.state.labels),
                    'class_balance': self.state.labels.mean(),
                }
            )

    def get_extractor(self):
        extr_class = getattr(extractors, self.extractor)
        return extr_class(None, self.n_jobs, self.device, self.verbose)

    def get_labels(self):
        '''
        Warnging: has sideeffect of mask modification
        '''
        markup = self.state.dataset.markup
        if self.labeling == 'hypomimia':
            labels = self.state.dataset.labels
        elif self.labeling in ('yury', 'mikhail'):
            labels = np.array(markup[self.labeling])
            labels[labels != 0] = 1
        elif self.labeling == 'coincide':
            # shrink default labels to coinciding
            labels = self.state.dataset.labels
            self.state.mask &= markup['mikhail'] == markup['yury']
        elif self.labeling == 'zero_coincide':
            # set 0 (healthy) only to coincided zeros
            labels = markup['mikhail'] + markup['yury']
            labels[labels != 0] = 1
        else:
            raise ValueError('Unknown labels type')

        return labels


if __name__ == "__main__":
    exp = Experiment()
