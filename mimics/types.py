from typing import Iterable, Union, Optional, List, Tuple
from pathlib import Path

import numpy as np


File = Union[Path, str]
Directory = Union[Path, str]
Shapes = np.ndarray  # shaped (#frames, #points, 2)
