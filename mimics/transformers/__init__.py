from sklearn.pipeline import make_pipeline

from ..types import Optional
from .basic_transformers import Transformer, Identical, Transposer, Flattener
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
    points: str = 'brows', *, steps: Optional[int] = None,
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
        ButterFilter(30.0, 0.1, 3.0, 4, preserve_mean=True),
        Centerer(ex.inds_68[points]),
        ChannelsSelector(ex.inds_68[points]),
    )
    return make_pipeline(*transformers[:steps])
