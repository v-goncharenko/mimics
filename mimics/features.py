import numpy as np


def windowed_correlation(shapes, r_points: tuple, l_points: tuple, win_len: int):
    '''Produces response of windowed correlation of a given window and points

    Result has same len as input shape (to be able to add it as a new channel to original data)
    '''
    right = shapes[:, r_points, :].mean(axis=1)
    left = shapes[:, l_points, :].mean(axis=1)
    l_indent, r_indent = -(win_len // 2), win_len - win_len // 2

    corrs = np.empty((len(shapes), 2))
    for i in range(len(shapes)):
        sli = slice(max(i + l_indent, 0), i + r_indent)
        corrs[i] = (
            np.corrcoef(right[sli, 0], left[sli, 0])[0, 1],
            np.corrcoef(right[sli, 1], left[sli, 1])[0, 1],
        )
    return corrs
