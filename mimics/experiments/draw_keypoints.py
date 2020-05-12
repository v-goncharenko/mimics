from dataclasses import dataclass

from joblib import Parallel, delayed

from ..datasets import FaceLandmarksDataset
from ..transformers import ConditionalFilter
from ..transformers import extractors as ex
from ..types import Directory, Tuple
from ..utils import data_dir
from ..visualizers import points_on_video


@dataclass
class DrawKeypoints(object):
    '''Draws different extractors' points on dataset's videos, saves to disk

    Extracted points are lowpass filtered for better human perception
    '''

    dataset_dir: Directory
    extractors: Tuple[str] = ('Dlib', 'Fa', 'San')
    cutoff: float = 2.5
    n_jobs: int = 1
    artifacts_dir: Directory = data_dir / 'tmp'
    verbose: bool = False

    @staticmethod
    def _draw(i, row, datasets, dataset_dir, save_dir, verbose):
        filename = dataset_dir / row['filename']
        save_file = save_dir / f'{filename.stem}_points.mp4'
        if save_file.exists():
            if verbose:
                print(f'{i} exists')
            return

        if verbose:
            print(f'{i} is generating')
        points_on_video(
            filename,
            {name: ds[i] for name, ds in datasets.items()},
            row['fps'],
            title=f'recording {i}',
            save_to=save_file,
        )

    def evaluate(self):
        save_dir = self.artifacts_dir / f'points_{self.dataset_dir.name}'
        save_dir.mkdir(parents=True, exist_ok=True)

        datasets = {
            name: FaceLandmarksDataset(
                self.dataset_dir,
                getattr(ex, f'{name}Extractor')(device='cpu'),
                ConditionalFilter((self.cutoff,), 4, 'lowpass'),
            )
            for name in self.extractors
        }
        markup = datasets[self.extractors[0]].markup

        Parallel(n_jobs=self.n_jobs)(
            delayed(self._draw)(
                i, row, datasets, self.dataset_dir, save_dir, self.verbose
            )
            for i, row in markup.iterrows()
        )
