import numpy as np
from sklearn.pipeline import make_pipeline

from ..types import Optional, Union
from . import extractors as ex
from .basic import Flattener, Identical, Transformer, Transposer
from .dataset_transformers import (
    ConditionalFilter,
    DatasetTransformer,
    Resampler,
    Stacker,
)
from .transformers import (
    ButterFilter,
    Centerer,
    ChannelsSelector,
    EyesRotator,
    PcaReducer,
    PointsToChannels,
    PositiveCorrelator,
    Scaler,
    Smoother,
    Stabilzer,
)


def get_preprocessing(
    points: Union[str, tuple] = 'brows',
    low: float = 0.2,
    high: float = 3.0,
    resample_to: Optional[float] = None,
    *,
    steps: Optional[int] = None,
    preserve_mean: bool = False
):
    '''Makes preprocessing pipeline of face shapes for classification

    Args:
        points: which points to center on and restrict output to
        resample_to: resulting rate of recording.
            By default equals to double high filtering rate according to Kotelnikov theorem
        steps: number of setps of final transform to make from 0 to max preprocessors
            (useful for debugging and visualising)
    '''
    if isinstance(points, str):
        points = (points,)
    resample_to = resample_to or 2 * high

    if steps == 0:
        return make_pipeline(Identical())

    transformers = (
        Stabilzer(ex.inds_68['nose']),
        EyesRotator(ex.inds_68['left_eye'], ex.inds_68['right_eye']),
        Scaler(ex.inds_68['left_eye'], ex.inds_68['right_eye'], axis=0),
        Scaler(ex.inds_68['eyes'], ex.inds_68['lips'], axis=1),
        ConditionalFilter((low, high), 4, preserve_mean=preserve_mean),
        Resampler(resample_to),
        Stacker(),
        ChannelsSelector(np.concatenate([ex.inds_68[po] for po in points])),
    )
    return make_pipeline(*transformers[:steps])


__all__ = (
    'DatasetTransformer',
    'Resampler',
    'Stacker',
    'ConditionalFilter',
    'Flattener',
    'Identical',
    'Transformer',
    'Transposer',
    'ButterFilter',
    'Centerer',
    'ChannelsSelector',
    'EyesRotator',
    'PcaReducer',
    'PositiveCorrelator',
    'Scaler',
    'Smoother',
    'Stabilzer',
    'PointsToChannels',
    'get_preprocessing',
)
