from typing import List

import numpy as np
from sklearn.decomposition import PCA
from scipy import signal

from .. import utils
from .basic_transformers import Transformer


class Stabilzer(Transformer):
    '''Subtracts mean of points with given indexes from each frame's points
        thus it stabilizer each frame wrt given points
    '''

    def __init__(self, indexes: tuple = tuple(range(27, 36))):
        self.indexes = indexes

    def transform(self, x):
        '''
        Args:
            x: list of sessions
        '''
        result = []
        self.centers = []
        for session in x:
            means = session[:, self.indexes, :].mean(axis=1, keepdims=True)
            result.append(session - means)
            self.centers.append(means)
        return result


class EyesRotator(Transformer):
    '''Rotates points to lay fixed points (brows) on horizontal line
    '''

    def __init__(self, left_indexes: tuple, right_indexes: tuple):
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

    def transform(self, x):
        result = []
        self.angles = []
        for session in x:
            means = session.mean(0)
            left = means[self.left_indexes, :].mean(0)
            right = means[self.right_indexes, :].mean(0)
            angle = utils.angle(np.array([-1, 0]), left - right)
            self.angles.append(angle)
            R = utils.rotation(-angle)
            result.append(session @ R.T)
        return result


class Scaler(Transformer):
    '''Scales mean of given points over axis to have given distance
    '''

    def __init__(self, inds1: tuple, inds2: tuple, axis: int, distance: float = 100.0):
        self.axis = axis
        self.inds1 = inds1
        self.inds2 = inds2
        self.distance = distance

    def transform(self, batch: list):
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


class Centerer(Transformer):
    '''Centers points such that means of given points are in (0, 0) point
    '''

    def __init__(self, inds: tuple):
        self.inds = inds

    def transform(self, batch: list):
        result = []
        for item in batch:
            means = item.mean(0)
            center = means[self.inds, :].mean(0)
            result.append(item - center)
        return result


class ChannelsSelector(Transformer):
    '''Filters face landmarks according to provided channels list'''

    def __init__(self, channels: tuple = tuple(range(17, 27))):
        self.channels = channels

    def transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        return [session[:, self.channels, :] for session in x]


class Smoother(Transformer):
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

    def transform(self, batch):
        window = getattr(np, self.window_type)(self.window_length) / self.window_length
        window /= window.sum()  # normalization to have same units as input
        return [
            np.apply_along_axis(np.convolve, 0, item, window, mode='valid')
            for item in batch
        ]


class PcaReducer(Transformer):
    '''Reduced less variational dimention of the points
    '''

    def __init__(self, n_components=1):
        self.n_components = n_components

    def transform(self, batch):
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


class PositiveCorrelator(Transformer):
    '''Makes all the channels in input positive correlated with zero channel

    Input have to be shaped (#channels, #samples)
    '''

    def transform(self, batch):
        result = []
        for item in batch:
            corrs = np.sign([np.corrcoef(item[0], channel)[0, 1] for channel in item])
            result.append(corrs[:, None] * item)
        return result


class ButterFilter(Transformer):
    '''Applies Butterworth filter bidirectionally

    more details https://scipy-cookbook.readthedocs.io/items/FiltFilt.html
    '''

    def __init__(
        self,
        rate: float,
        low: float,
        high: float,
        order: int = 4,
        *,
        preserve_mean: bool = False,
    ) -> None:
        self.rate = rate
        self.low = low
        self.high = high
        self.order = order
        self.preserve_mean = preserve_mean

        self.design = utils.butter_design(low, high, rate, order, 'bandpass')

    def transform(self, batch):
        return [
            signal.filtfilt(*self.design, session, axis=0)
            + (session.mean(0, keepdims=True) if self.preserve_mean else 0)
            for session in batch
        ]
