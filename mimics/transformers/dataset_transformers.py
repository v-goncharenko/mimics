from functools import partial

import numpy as np
from scipy import interpolate, signal
from torch.utils.data import Dataset

from .. import utils
from ..types import Tuple
from .basic import Transformer


class DatasetTransformer(Transformer):
    '''Special transformer that takes whole dataset as an input
        This tranformers mutate given dataset

    Using this type is required when transform of each dataset item
        (face landmarks sequences) depends on other information in dataset.
    By default descendants is assumed regular transformers impolementing `_transform`
        method and not require whole dataset, just data.
    In case of transformer truly dependant on the whole dataset one have to redefine
        `transform` method itself
    '''

    def transform(self, data):
        '''
        Args:
            dataset: one object of dataset class
        '''
        if isinstance(data, Dataset):
            data.data = self._transform(data.data)
        else:
            data = self._transform(data)
        return data

    def _transform(self, batch):
        raise NotImplementedError()


class Resampler(DatasetTransformer):
    '''Resamples signals in a given dataset to one rate.

    Method implemented uses local linear interpolation.
    Alternatives are scipy.signal's resample and resample_poly.
    Former gives high freq artifacts on start and sharp edges.
    Latter requires searching for best fraction to estimate target_rate which is hard.
    The only shortcoming of interpolation is poor handling of high freqs (above target),
        this colud be solved by lowpass filter before resampling (noe it is not done yet).
    Finally tests showed that linear interpolation works fine on our data
        in range from 6 to 30 Hz to 15 Hz
    One shortcoming of this is phase shift in case of big frequency difference

    Later I used scipy's interpolate module and that's wearpons of victory.
        No disadvantages observed
    '''

    kind_np = 'np'
    kind_scipy = 'scipy'

    @staticmethod
    def resample_interp(signal: np.ndarray, in_rate: float, out_rate: float):
        '''Resamples by interpolation

        copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py
        '''
        new_len = round(len(signal) / in_rate * out_rate)

        return np.interp(
            np.linspace(0.0, 1.0, new_len),  # where to interpret
            np.linspace(0.0, 1.0, len(signal)),  # known positions
            signal,  # known data points
        )

    @staticmethod
    def resample_interp_mult(signal, in_rate: float, out_rate: float, axis: int):
        funct1d = partial(Resampler.resample_interp, in_rate=in_rate, out_rate=out_rate)
        return np.apply_along_axis(funct1d, axis, signal)

    @staticmethod
    def resample_scipy(signal, in_rate: float, out_rate: float, axis: int):
        funct = interpolate.interp1d(
            np.linspace(0.0, 1.0, len(signal)),
            signal,
            axis=axis,
            copy=False,
            assume_sorted=True,
        )
        new_len = round(len(signal) / in_rate * out_rate)
        return funct(np.linspace(0.0, 1.0, new_len))

    def __init__(self, target_rate: float, *, kind: str = kind_scipy):
        self.target_rate = target_rate
        self.kind = kind

    def transform(self, dataset):
        if self.kind == self.kind_scipy:
            funct = self.resample_scipy
        else:
            funct = self.resample_interp_mult

        dataset.data = [
            funct(record, fps, self.target_rate, 0)
            for record, fps in zip(dataset, dataset.markup['fps'])
        ]
        return dataset


class ConditionalFilter(DatasetTransformer):
    '''Applies digital bandpass filter to signals in a dataset
        according to each signal's rate
    '''

    def __init__(
        self, cutoffs: Tuple[float], order: int, *, preserve_mean: bool = False,
    ):
        self.cutoffs = cutoffs
        self.order = order
        self.preserve_mean = preserve_mean

    def transform(self, dataset):
        for i, fps in enumerate(dataset.markup['fps']):
            if all(2 * cut < fps for cut in self.cutoffs):
                btype = 'bandpass'
                cutoffs = self.cutoffs
            else:
                btype = 'highpass'
                cutoffs = (min(self.cutoffs),)
            design = utils.butter_design(fps, cutoffs, self.order, btype)

            data = dataset.data[i]
            dataset.data[i] = signal.filtfilt(*design, data, axis=0)
            if self.preserve_mean:
                dataset.data[i] += data.mean(0, keepdims=True)
        return dataset


class Stacker(DatasetTransformer):
    def transform(self, dataset):
        dataset.data = np.stack(dataset.data)
        return dataset
