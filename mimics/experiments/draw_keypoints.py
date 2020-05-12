from dataclasses import dataclass

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
    artifacts_dir: Directory = data_dir / 'tmp'
    verbose: bool = False

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

        for i, row in markup.iterrows():
            filename = self.dataset_dir / row['filename']
            save_file = save_dir / f'{filename.stem}_points.mp4'
            if save_file.exists():
                if self.verbose:
                    print(f'{i} exists')
                continue

            if self.verbose:
                print(f'{i} is generating')
            points_on_video(
                filename,
                {name: ds[i] for name, ds in datasets.items()},
                row['fps'],
                title=f'recording {i}',
                save_to=save_file,
            )
            break
