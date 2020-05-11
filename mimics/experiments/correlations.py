import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow

from ..features import windowed_correlation
from ..transformers.extractors import inds_68
from ..types import Device, Directory
from ..utils import data_dir
from .base import BaseExperiment


@dataclass
class WindowedCorrelations(BaseExperiment):
    '''Plots windowed correlations for points

    Note: only Y coordinate correlations are plotted
    '''

    device: Device = 'cpu'
    windows: tuple = (1, 1.5, 2)  # seconds
    axes: tuple = (0, 1)
    right_points: tuple = inds_68['right_brow']
    left_points: tuple = inds_68['left_brow']
    artifacts_dir: Directory = data_dir / 'tmp'

    axes_names = {0: 'X', 1: 'Y'}

    def __post_init__(self):
        if self.points != 'all':
            warnings.warn("Points are not 'all', default numbering may be broken")

    def evaluate(self):
        super().evaluate()
        self.artifacts_dir.mkdir(exist_ok=True)

        markup = self.state.dataset.markup[self.state.mask]
        plt.figure(constrained_layout=True)

        with mlflow.start_run():
            self.log_run()

            for i, shapes in enumerate(self.state.features):
                orig_id = markup.iloc[i]['id']
                filename = self.artifacts_dir / f"{orig_id:03} correlations.png"
                if self.verbose:
                    print(filename.stem)

                plt.clf()
                for win in self.windows:
                    win_len = int(round(win * markup.iloc[i]['fps']))
                    win_cor = windowed_correlation(
                        shapes, self.right_points, self.left_points, win_len
                    )
                    for axis in self.axes:
                        plt.plot(
                            win_cor[:, axis],
                            label=f'{self.axes_names[axis]}, window {win} sec',
                        )
                plt.legend()
                plt.title(filename.stem)
                plt.savefig(filename)
                mlflow.log_artifact(filename)
