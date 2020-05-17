from dataclasses import dataclass
from itertools import count

import matplotlib.pyplot as plt
import mlflow
from mne.decoding import CSP

from ..transformers import PointsToChannels
from ..types import Device, Directory
from ..utils import data_dir
from .base import BaseExperiment


@dataclass
class CrossvalidatedCsp(BaseExperiment):
    '''Plots crossvalidated CSP extracted signal for each record in dataset
    '''

    device: Device = 'cpu'
    artifacts_dir: Directory = data_dir / 'tmp'

    def evaluate(self):
        super().evaluate()
        self.artifacts_dir.mkdir(exist_ok=True)

        names = ('both', 'X', 'Y')
        signals = tuple(
            PointsToChannels().fit_transform(self.state.features[..., sli])
            for sli in (slice(None), slice(None, 1), slice(1, None))
        )
        labels = self.state.labels
        markup = self.state.dataset.markup[self.state.mask]
        with mlflow.start_run():
            self.log_run()

            plt.figure(figsize=(10, 7), constrained_layout=True)
            for train_inds, test_inds in self.state.cv.split(signals[0], labels):
                mixed = []
                for signal in signals:
                    csp = CSP(n_components=1, transform_into='csp_space')
                    csp.fit(signal[train_inds], labels[train_inds])
                    # normalizing filters
                    csp.filters_ = csp.filters_ / csp.filters_.sum(axis=1, keepdims=True)
                    mixed.append(csp.transform(signal[test_inds])[:, 0, :])

                for test_ind, markup_ind in enumerate(test_inds):
                    orig_id = markup.iloc[markup_ind]['id']
                    filename = self.artifacts_dir / f'{orig_id:03} record CSP.png'
                    if self.verbose:
                        print(filename.stem)

                    plt.clf()
                    for i, name, mix in zip(count(), names, mixed):
                        plt.plot(mix[test_ind] + 20 * i, label=name)
                    plt.title(filename.stem)
                    plt.legend(
                        loc='lower right',
                        frameon=True,
                        facecolor='w',
                        edgecolor='r',
                        framealpha=0.6,
                    )
                    plt.savefig(filename)
                    mlflow.log_artifact(filename)
