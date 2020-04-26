from functools import partial

import numpy as np
from scipy import signal

from .. import datasets, utils
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
        if isinstance(data, datasets.FaceLandmarksDataset):
            data._data = self._transform(data._data)
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
    '''

    @staticmethod
    def resample_interp(
        signal: np.ndarray, in_rate: float, out_rate: float, endpoint: bool = False
    ):
        '''Resamples by interpolation

        copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py
        '''
        scale = out_rate / in_rate
        new_len = round(len(signal) * scale)

        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        return np.interp(
            np.linspace(0.0, 1.0, new_len, endpoint=endpoint),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=endpoint),  # known positions
            signal,  # known data points
        )

    @staticmethod
    def resample_interp_mult(
        signal, in_rate: float, out_rate: float, axis: int, endpoint: bool = False
    ):
        funct1d = partial(Resampler.resample_interp, in_rate=in_rate, out_rate=out_rate)
        return np.apply_along_axis(funct1d, axis, signal)

    def __init__(self, target_rate: float, endpoint: bool = False):
        self.target_rate = target_rate
        self.endpoint = endpoint

    def transform(self, dataset):
        dataset._data = [
            self.resample_interp_mult(record, fps, self.target_rate, 0, self.endpoint)
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
            if all(cut < 2 * fps for cut in self.cutoffs):
                btype = 'bandpass'
                cutoffs = self.cutoffs
            else:
                btype = 'highpass'
                cutoffs = min(self.cutoffs)
            design = utils.butter_design(fps, cutoffs, self.order, btype)

            data = dataset._data[i]
            dataset._data[i] = signal.filtfilt(*design, data, axis=0)
            if self.preserve_mean:
                dataset._data[i] += data.mean(0, keepdims=True)
        return dataset
