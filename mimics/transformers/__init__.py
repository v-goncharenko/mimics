from sklearn.pipeline import make_pipeline

from ..types import Optional
from .basic import Transformer, Identical, Transposer, Flattener
from .dataset_transformers import DatasetTransformer, Resampler
from .transformers import (
    Stabilzer,
    EyesRotator,
    Scaler,
    Centerer,
    ChannelsSelector,
    Smoother,
    PcaReducer,
    PositiveCorrelator,
    ButterFilter,
)
from . import extractors as ex


def get_preprocessing(
    points: str = 'brows',
    resample_to: float = 15.0,
    low: float = 0.1,
    high: float = 5.0,
    *,
    steps: Optional[int] = None,
    preserve_mean: bool = False
):
    '''Makes preprocessing pipeline for face shapes

    Args:
        points: which points to center on and restrict output to
        steps: number of setps of final transform to make from 0 to max preprocessors
            (useful for debugging and visualising)
    '''
    if steps == 0:
        return make_pipeline(Identical())

    transformers = (
        Stabilzer(ex.inds_68['nose']),
        EyesRotator(ex.inds_68['left_eye'], ex.inds_68['right_eye']),
        Scaler(ex.inds_68['left_eye'], ex.inds_68['right_eye'], axis=0),
        Scaler(ex.inds_68['eyes'], ex.inds_68['lips'], axis=1),
        Resampler(resample_to),
        ButterFilter(resample_to, (low, high), 4, preserve_mean=preserve_mean),
        Centerer(ex.inds_68[points]),
        ChannelsSelector(ex.inds_68[points]),
    )
    return make_pipeline(*transformers[:steps])
