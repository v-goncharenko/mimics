import numpy as np
from scipy import signal
from sklearn.decomposition import PCA

from .. import utils
from ..types import List, Tuple
from .dataset_transformers import DatasetTransformer


class Stabilzer(DatasetTransformer):
    '''Subtracts mean of points with given indexes from each frame's points
        thus it stabilizer each frame wrt given points
    '''

    def __init__(self, indexes: tuple):
        self.indexes = indexes

    def _transform(self, batch):
        '''
        Args:
            batch: list of sessions
        '''
        result = []
        self.centers = []
        for session in batch:
            means = session[:, self.indexes, :].mean(axis=1, keepdims=True)
            result.append(session - means)
            self.centers.append(means)
        return result


class EyesRotator(DatasetTransformer):
    '''Rotates points to lay fixed points (brows) on horizontal line
    '''

    def __init__(self, left_indexes: tuple, right_indexes: tuple):
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

    def _transform(self, batch):
        result = []
        self.angles = []
        for session in batch:
            means = session.mean(0)
            left = means[self.left_indexes, :].mean(0)
            right = means[self.right_indexes, :].mean(0)
            angle = utils.angle(np.array([-1, 0]), left - right)
            self.angles.append(angle)
            R = utils.rotation(-angle)
            result.append(session @ R.T)
        return result


class Scaler(DatasetTransformer):
    '''Scales mean of given points over axis to have given distance
    '''

    def __init__(self, inds1: tuple, inds2: tuple, axis: int, distance: float = 100.0):
        self.axis = axis
        self.inds1 = inds1
        self.inds2 = inds2
        self.distance = distance

    def _transform(self, batch: list):
        result = []
        self.scales = []
        for item in batch:
            means = item.mean(0)
            obj1 = means[self.inds1, :].mean(0)
            obj2 = means[self.inds2, :].mean(0)
            alpha = self.distance / np.abs(obj1 - obj2)[self.axis]
            self.scales.append(alpha)
            scaled = item.copy()
            scaled[..., self.axis] *= alpha
            result.append(scaled)
        return result


class Centerer(DatasetTransformer):
    '''Centers points such that means of given points are in (0, 0) point
    '''

    def __init__(self, inds: tuple):
        self.inds = inds

    def _transform(self, batch: list):
        result = []
        for item in batch:
            means = item.mean(0)
            center = means[self.inds, :].mean(0)
            result.append(item - center)
        return result


class ChannelsSelector(DatasetTransformer):
    '''Filters face landmarks according to provided channels list'''

    def __init__(self, channels: tuple):
        self.channels = channels

    def _transform(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        return [session[:, self.channels, :] for session in batch]


class Smoother(DatasetTransformer):
    '''Smooths given batch of signals with given window via convolution
    '''

    def __init__(
        self, window_length: int, window_type: str = 'hamming', mode: str = 'valid'
    ):
        '''
        Args:
            window_type: can be found at https://docs.scipy.org/doc/numpy/reference/routines.window.html
            mode: as in np.convolve: valid, same, full
        '''
        self.window_length = window_length
        self.window_type = window_type

    def _transform(self, batch):
        window = getattr(np, self.window_type)(self.window_length) / self.window_length
        window /= window.sum()  # normalization to have same units as input
        return [
            np.apply_along_axis(np.convolve, 0, item, window, mode='valid')
            for item in batch
        ]


class PcaReducer(DatasetTransformer):
    '''Reduced less variational dimention of the points
    '''

    def __init__(self, n_components=1):
        self.n_components = n_components

    def _transform(self, batch):
        pca = PCA(self.n_components)
        return [
            np.squeeze(
                np.stack(
                    [
                        pca.fit_transform(item[:, point, :])
                        for point in range(item.shape[1])
                    ],
                    1,
                )
            )
            for item in batch
        ]


class PositiveCorrelator(DatasetTransformer):
    '''Makes all the channels in input positive correlated with zero channel

    Input have to be shaped (#channels, #samples)
    '''

    def _transform(self, batch):
        result = []
        for item in batch:
            corrs = np.sign([np.corrcoef(item[0], channel)[0, 1] for channel in item])
            result.append(corrs[:, None] * item)
        return result


class ButterFilter(DatasetTransformer):
    '''Applies Butterworth filter bidirectionally

    more details https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
    '''

    def __init__(
        self,
        rate: float,
        cutoffs: Tuple[float],
        order: int = 4,
        btype: str = 'bandpass',
        *,
        preserve_mean: bool = False,
    ):
        self.rate = rate
        self.cutoffs = cutoffs
        self.order = order
        self.btype = btype
        self.preserve_mean = preserve_mean

        self.design = utils.butter_design(rate, cutoffs, order, btype)

    def _transform(self, batch):
        return [
            signal.filtfilt(*self.design, session, axis=0)
            + (session.mean(0, keepdims=True) if self.preserve_mean else 0)
            for session in batch
        ]
