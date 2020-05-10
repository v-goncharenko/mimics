import matplotlib.pyplot as plt
import mlflow
from mne.decoding import CSP

from ..transformers import PointsToChannels
from ..types import Directory
from ..utils import data_dir
from .base import BaseExperiment


class CrossvalidatedCsp(BaseExperiment):
    '''Plots crossvalidated CSP extracted signal for each record in dataset
    '''

    artifacts_dir: Directory = data_dir / 'tmp'

    def evaluate(self):
        super().evaluate()
        self.artifacts_dir.mkdir(exist_ok=True)

        channelled = PointsToChannels().fit_transform(self.state.features)
        labels = self.state.labels
        markup = self.state.dataset.markup[self.state.mask]
        with mlflow.start_run():
            self.log_run()

            plt.figure(constrained_layout=True)
            for train_inds, test_inds in self.state.cv.split(channelled, labels):
                csp = CSP(n_components=1, transform_into='csp_space')
                csp.fit(channelled[train_inds], labels[train_inds])
                signals = csp.transform(channelled[test_inds])[:, 0, :]
                for ind, signal in zip(test_inds, signals):
                    orig_id = markup.iloc[ind, 0]
                    filename = self.artifacts_dir / f'CSP record {orig_id:03}.png'
                    if self.verbose:
                        print(filename.stem)

                    plt.clf()
                    plt.plot(signal)
                    plt.title(filename.stem)
                    plt.savefig(filename)
                    mlflow.log_artifact(filename)
