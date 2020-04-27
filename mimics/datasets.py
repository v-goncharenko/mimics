import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .transformers import DatasetTransformer
from .transformers import extractors as extrs
from .types import Directory, File, Optional
from .visualizers import points_on_video


class FaceLandmarksDataset(Dataset):
    '''Dataset contains timeseries of face landmarks from videos
        Also contains markup DataFrame with metainformation about records

    Markup have to have 'filename' column with name of videofile
    '''

    markup_filename = 'markup.csv'
    precomputed_dir = 'precomputed'

    def __init__(
        self,
        path: Directory,
        extractor: extrs.VideoLandmarksExtractor,
        transformer: Optional[DatasetTransformer] = None,
        *,
        compute_fps: bool = True,
    ):
        '''
        Args:
            compute_fps: if True drops fps from markup and computes fps as
                actual frames number divided by record duration (from markup).
                Else uses values from markup as is
        '''
        self.path = path
        self.extractor = extractor
        self.transformer = transformer

        self.markup = pd.read_csv(self.path / self.markup_filename)

        precomp_path = (
            self.path / self.precomputed_dir / f'{extractor.__class__.__name__}.pickle'
        )
        if precomp_path.exists():
            with open(precomp_path, 'rb') as file:
                self.data = pickle.load(file)
        else:
            self.data = self.extractor.fit_transform(
                [self.path / filename for filename in self.markup['filename']]
            )
            precomp_path.parent.mkdir(exist_ok=True)
            with open(precomp_path, 'wb') as file:
                pickle.dump(self.data, file)

        if compute_fps:
            frames = np.array([len(rec) for rec in self.data])
            self.markup['fps'] = frames / self.markup['duration']

        if self.transformer:
            self.transformer.fit_transform(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return (
            f'FaceLandmarksDataset of {len(self)} records\n'
            f'extracted by {self.extractor.__class__.__name__}\n'
            f'minimal shape {min(self, key=lambda record: record.shape).shape}'
        )

    @property
    def labels(self):
        return self.markup['hypomimia'].values

    def filename(self, index: int):
        return self.path / self.markup.loc[index, 'filename']

    def fps(self, index: int):
        return self.markup.loc[index, 'fps']

    def video(self, index: int, *, html5: bool = True, save_to: Optional[File] = None):
        return points_on_video(
            self.filename(index),
            self[index],
            self.fps(index),
            html5=html5,
            save_to=save_to,
        )
