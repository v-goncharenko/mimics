from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):
    '''Base class for transformers providing dummy implementation
        of the methods expected by sklearn
    '''
    def fit(self, x, y=None):
        return self


class Identical(Transformer):
    def transform(self, batch):
        return batch


class Transposer(Transformer):
    '''Transponses every item in input (useful for changing main axis in data)
    '''
    def transform(self, batch):
        return [item.T for item in batch]


class Flattener(Transformer):
    '''
    Args:
        batch: numpy.ndarray - usually `_data` of shrinked dataset
    '''
    def transform(self, batch):
        return batch.reshape((len(batch), -1))
