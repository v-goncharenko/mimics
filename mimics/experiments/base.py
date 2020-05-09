from collections import namedtuple
from dataclasses import dataclass

import mlflow
from sklearn.model_selection import StratifiedKFold

from ..datasets import FaceLandmarksDataset
from ..transformers import extractors, get_preprocessing
from ..types import ClassVar, Device, Directory, Tuple, Union
from ..utils import default_device


@dataclass
class BaseExperiment(object):
    '''
    Args:
        name: name of the experiment for MLflow
        extractor: classname of Extractor see `.tranformers.extractors`
        points: param for `get_preprocessing`
        cutoffs: tuple of low and high frequencies to filter signals
        exercise: 'B' for brows, 'S' for smile
        labeling: see `get_labels` method
        cv: number of folds in crossvalidation
        n_jobs: parallelization param for joblib
        device: torch device to use
        verbose: printing to console or no
        log: enable MLflow logging (disabling is handy for debugging)
    '''

    name: str
    dataset_dir: Directory
    extractor: str
    points: Union[str, tuple]
    cutoffs: Tuple[float, float]
    exercise: str
    labeling: str
    cv: int = 5
    n_jobs: int = 1
    device: Device = default_device
    verbose: bool = False
    log: bool = True

    state_fields: ClassVar[str] = ''

    def get_extractor(self):
        extr_class = getattr(extractors, self.extractor)
        return extr_class(None, self.n_jobs, self.device, self.verbose)

    def evaluate(self):
        if self.log:
            mlflow.set_experiment(self.name)
        self.state = namedtuple('State', self.state_fields)

        self.state.cv = StratifiedKFold(self.cv, True)

        extr = self.get_extractor()
        preproc = get_preprocessing(self.points, *self.cutoffs)
        self.state.dataset = FaceLandmarksDataset(self.dataset_dir, extr, preproc)

        self.state.mask = self.state.dataset.markup['exercise'] == self.exercise
        self.state.features = self.state.dataset.data[self.state.mask]
        self.state.labels = self.state.dataset.labels(self.labeling)[self.state.mask]
