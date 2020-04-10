import pickle
import json

import numpy as np
import pandas as pd
from torch.utils import data as torch_data

from .extractors import VideoFaceLandmarksExtractor
from .transformers import Transformer
from .types import Directory, Optional, File
from .visualizers import points_on_video


class FaceLandmarksDataset(torch_data.Dataset):
    '''Provides markup data for objects which are timeseries of points
        represented by videos on disk

    Markup have to have 'filename' column with name of videofile
    '''
    markup_filename = 'markup.csv'
    precomputed_dir = 'precomputed'

    def __init__(
        self,
        path: Directory,
        extractor: VideoFaceLandmarksExtractor,
        transformer: Transformer=None,
    ):
        self.path = path
        self.extractor = extractor
        self.transformer = transformer

        self.markup = pd.read_csv(self.path/self.markup_filename)

        precomp_path = self.path/self.precomputed_dir/f'{extractor.__class__.__name__}.pickle'
        if precomp_path.exists():
            with open(precomp_path, 'rb') as file:
                self._data = pickle.load(file)
        else:
            self._data = self.extractor.fit_transform(
                [self.path/filename for filename in self.markup['filename']]
            )
            precomp_path.parent.mkdir(exist_ok=True)
            with open(precomp_path, 'wb') as file:
                pickle.dump(self._data, file)

        if transformer:
            self._data = self.transformer.fit_transform(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

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
        return self.path/self.markup.loc[index, 'filename']

    def fps(self, index: int):
        return self.markup.loc[index, 'fps']

    def video(self, index: int, *, html5: bool=True, save_to: Optional[File]=None):
        return points_on_video(
            self.filename(index),
            self[index],
            self.fps(index),
            html5=html5,
            save_to=save_to,
        )
