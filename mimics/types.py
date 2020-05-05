from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch


File = Union[Path, str]
Directory = Union[Path, str]
Shapes = np.ndarray  # shaped (#frames, #points, 2)
Device = Union[str, torch.device]


__all__ = (
    'File',
    'Directory',
    'Shapes',
    'Device',
    'Iterable',
    'List',
    'Optional',
    'Tuple',
    'Union',
)
