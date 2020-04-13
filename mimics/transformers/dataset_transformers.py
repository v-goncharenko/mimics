from functools import partial

import numpy as np

from .basic_transformers import Transformer


class DatasetTransformer(Transformer):
    '''Special transformer that takes whole dataset as an input
        This tranformers mutate given dataset
    '''

    def transform(self, dataset):
        '''
        Args:
            dataset: one object of dataset class
        '''
        raise NotImplementedError()


class Resampler(DatasetTransformer):
    '''Resamples signals in a given dataset to one rate.

    Method implemented uses local linear interpolation.
    Alternatives are scipy.signal's resample and resample_poly.
    Former gives high freq artifacts on start and sharp edges.
    Latter requires searching for best fraction to estimate target_rate which is hard.

    Finally tests showed that linear interpolation works fine on our data
        in range from 6 to 30 Hz to 15 Hz
    '''

    @staticmethod
    def resample_interp(signal, input_fs, output_fs):
        '''copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py
        '''
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)

        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
        return resampled_signal

    @staticmethod
    def resample_interp_mult(signal, in_rate, out_rate, axis):
        funct1d = partial(Resampler.resample_interp, input_fs=in_rate, output_fs=out_rate)
        return np.apply_along_axis(funct1d, axis, signal)

    def __init__(self, target_rate: float):
        self.target_rate = target_rate

    def transform(self, dataset):
        return [
            self.resample_interp_mult(record, fps, self.target_rate, 0)
            for record, fps in zip(dataset, dataset.markup['fps'])
        ]
