from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


File = Union[Path, str]
Directory = Union[Path, str]
Shapes = np.ndarray  # shaped (#frames, #points, 2)


__all__ = (
    'File',
    'Directory',
    'Shapes',
    'Iterable',
    'List',
    'Optional',
    'Tuple',
    'Union',
)
