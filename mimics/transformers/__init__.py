from sklearn.pipeline import make_pipeline

from ..types import Optional
from .. import extractors as ex
from .basic_transformers import Transformer, Identical, Transposer, Flattener
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


def get_preprocessing(
    points: str = 'brows',
    *,
    steps: Optional[int] = None,
    window_length: int = 7,  # emphirical default for 30 fps
):
    '''Makes preprocessing pipeline for face shapes

    Args:
        steps: number of setps of final transform to make (useful for debugging and visualising)
    '''
    if steps == 0:
        return make_pipeline(Identical())

    transformers = (
        Stabilzer(ex.inds_68['nose']),
        EyesRotator(ex.inds_68['left_eye'], ex.inds_68['right_eye']),
        Scaler(ex.inds_68['left_eye'], ex.inds_68['right_eye'], axis=0),
        Scaler(ex.inds_68['eyes'], ex.inds_68['lips'], axis=1),
        Centerer(ex.inds_68[points]),
        Smoother(window_length, 'hamming'),
        ChannelsSelector(ex.inds_68[points]),
    )
    return make_pipeline(*transformers[:steps])
