from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Union, Tuple

import cv2
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.patches as patches


@contextmanager
def open_video(video_path: Union[Path, str], mode: str = 'r', *args):
    '''Context manager to work with cv2 videos
        Mimics python's standard `open` function

    Args:
        video_path: path to video to open
        mode: either 'r' for read or 'w' write
        args: additional arguments passed to Capture or Writer
            according to OpenCV documentation
    Returns:
        cv2.VideoCapture or cv2.VideoWriter depending on mode

    Example of writing:
        open_video(
            out_path,
            'w',
            cv2.VideoWriter_fourcc(*'XVID'), # fourcc
            15, # fps
            (width, height), # frame size
        )
    '''
    video_path = Path(video_path)
    if mode == 'r':
        video = cv2.VideoCapture(video_path.as_posix(), *args)
    elif mode == 'w':
        video = cv2.VideoWriter(video_path.as_posix(), *args)
    else:
        raise ValueError(f'Incorrect open mode "{mode}"; "r" or "w" expected!')
    if not video.isOpened():
        raise ValueError(f'Video {video_path} is not opened!')
    try:
        yield video
    finally:
        video.release()


def frames(
    video: Union[Path, str, cv2.VideoCapture], rgb: bool = True
) -> Iterable[np.ndarray]:
    '''Generator of frames of the video provided

    Args:
        video: either Path or Video capture to read frames from
            in former case file will be opened with :py:funct:`.open_video`
        rgb: if True returns RGB image, else BGR - native to opencv format
    Yields:
        Frames of video in (H, W, C) format
    '''
    if isinstance(video, Path) or isinstance(video, str):
        with open_video(video) as capture:
            yield from frames(capture, rgb)
    else:
        while True:
            retval, frame = video.read()
            if not retval:
                break
            if rgb:
                frame = frame[:, :, ::-1]
            yield frame


def get_meta(video_path: Path):
    '''Extracts main video meta data as dict

    Eliminates a need in ugly openCV constants

    Warning: can be long to execute on big files cause tries to count frames itself
    '''
    with open_video(video_path) as video:
        real_frame_count = sum(1 for _ in frames(video))
        return {
            'width': video.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': video.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': video.get(cv2.CAP_PROP_FPS),
            'fourcc': video.get(cv2.CAP_PROP_FOURCC),
            'frame_count_meta': video.get(cv2.CAP_PROP_FRAME_COUNT),
            'frame_count': real_frame_count,
        }


def rm_r(folder):
    for path in folder.glob('*'):
        if path.is_dir():
            rm_r(path)
        else:
            path.unlink()
    folder.rmdir()


def convert_bbox(bbox: tuple, fr: str, to: str) -> tuple:
    '''
    Converts bounding box from one fromat to other
    Available formats:
        * 'xywh' - top left point and width, height
        * 'tlbr' - top left point (x, y) bottom right point (x, y)
        * 'dlib' - dlib's rectangle.
    Note: make enum for `fr`, `to`
    '''
    if fr == 'xywh' and to == 'tlbr':
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    elif fr == 'tlbr' and to == 'xywh':
        l, t, r, b = bbox
        return [l, t, r - l, b - t]
    elif fr == 'dlib' and to == 'tlbr':
        return (bbox.left(), bbox.top(), bbox.right(), bbox.bottom())

    raise NotImplementedError('sorry, this functionality is not currently available')


def plot_image(
    image,
    title: str = '',
    *,
    figsize: tuple = (20, 5),
    boxes: list = [],
    opencv=True,
    extra_operations=lambda: None,
):
    '''
    boxes - list of bboxes in 'tlbr' format
        remember that matplotlib's coordinates x is horizontal, y is vertical
    extra_operations - lambda with everything you want to do to plt
    '''
    if opencv:  # to reverse colours from BGR
        image = image[..., ::-1]

    plt.figure(figsize=figsize)
    plt.imshow(image,)
    plt.title(title)
    plt.xlabel('Y (first coordinate)')
    plt.ylabel('X (second coordinate)')
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        plt.gca().add_patch(rect)
    extra_operations()
    plt.show()


def rotation(angle: float, radians: bool = True):
    '''Constructs rotation matirx in 2d space
    '''
    if not radians:
        angle = np.radians(angle)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((-c, s), (s, c)))


def unit_vector(vector):
    """ Returns the unit vector of the vector"""
    return vector / np.linalg.norm(vector)


def angle(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        raise NotImplementedError('Too odd vectors =(')
    return np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def butter_design(
    fs: float, cutoffs: Tuple[float], order: int = 4, btype: str = 'bandpass'
):
    '''Get Butterworth filter design with params specified

    implementation taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    '''
    nyq = 0.5 * fs
    normal = tuple(cut / nyq for cut in cutoffs)
    return signal.butter(order, normal, btype=btype)
