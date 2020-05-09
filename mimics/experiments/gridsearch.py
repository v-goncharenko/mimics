from dataclasses import dataclass

import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_validate

from ..types import ClassVar, Optional
from ..utils import data_dir
from .base import BaseExperiment


default_artifacts_dir = data_dir / 'tmp'


@dataclass
class GridSearch(BaseExperiment):
    '''
    Args:
        clfs: dict of classifiers and their params e.g. `.classifiers.clfs`
        scores: tuple of scores e.g. `.classifiers.scores`
    '''

    clfs: Optional[dict] = None
    scores: Optional[tuple] = None

    state_fields: ClassVar[str] = 'dataset cv mask labels features'

    def evaluate(self):
        super().evaluate()
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
                        f'{aggr.__name__}_{name}': aggr(cv_res[f'test_{name}']) or np.nan
                        for name in self.scores
                    }
                )
            mlflow.log_metrics(
                {
                    'records': len(self.state.labels),
                    'class_balance': self.state.labels.mean(),
                }
            )


# @dataclass
# class CrossvalidatedCsp(object):
