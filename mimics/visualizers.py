from pathlib import Path

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from .types import File, Iterable, Optional, Shapes, Union
from .utils import frames, open_video


def plot_joint(dataset, title='', *, orient_up: bool = True, figsize=(5, 5)):
    '''Plots first frame of each record of the dataset

    Args:
        orient_up: whether shaped in dataset oriented up (if False, rotates view)
    '''
    plt.figure(figsize=figsize)
    plt.title(title)

    rotation = 1 if orient_up else -1

    for record in dataset:
        plt.scatter(record[0, :, 0], rotation * record[0, :, 1])
    plt.show()


def anim2html(anim, html5: bool = True):
    html_video = anim.to_html5_video() if html5 else anim.to_jshtml()
    plt.close()  # this is to prevent notebook from displaying animation twice
    return HTML(html_video)


def shapes_animation(
    shapes,
    interval: float,
    *,
    orient_up: bool = True,
    title: str = '',
    figsize: tuple = (5, 5),
    html5: bool = True,
):
    '''Makes html display element of given shapes animation
    Args:
        orient_up: whether shaped in dataset oriented up (if False, rotates view)
            Implemented chutchy here =)
        interval: see FuncAnimation. Usually `1000 / fps` works well
        html5: to use html5 based video (doesn't supported by some browsers)
            if False - use js based video. it has more fancy player and are created twise longer
    '''
    figure = plt.figure(figsize=figsize)

    plt.title(title)

    mins, maxes = shapes.min(axis=(0, 1)), shapes.max(axis=(0, 1))
    plt.xlim(mins[0], maxes[0])
    if orient_up:
        plt.ylim(mins[1], maxes[1])
    else:
        plt.ylim(maxes[1], mins[1])

    line = plt.plot(shapes[0, :, 0], shapes[0, :, 1], '.')[0]

    def update_line(step: int):
        line.set_data(shapes[step, :, 0], shapes[step, :, 1])
        return (line,)

    anim = animation.FuncAnimation(
        figure, update_line, len(shapes), interval=interval, blit=True,
    )

    return anim2html(anim, html5)


def points_on_video(
    video: File,
    shapes: Union[dict, Shapes],
    fps: float,
    *,
    title: str = '',
    figsize: tuple = (10, 7),
    html5: bool = True,
    save_to: Optional[File] = None,
):
    '''Draws points from `shapes` on `video` and returns either html animation
        or saves animation on disk
    Args:
        video: video file to lay points to
        shapes: item of `dataset` corresponding to given video
        fps: frames per second in video
        html5: see `anim2html`
        save_to: path to save video to. If None - animation is returned
    '''
    if isinstance(shapes, dict):
        named_shapes = shapes
        legend = True
    else:
        named_shapes = {'': shapes}
        legend = False
    interval = 1000 / fps

    figure = plt.figure(figsize=figsize, constrained_layout=True)
    image = plt.imshow(next(iter(frames(video))))
    lines = []
    for name, shapes in named_shapes.items():
        lines.append(plt.plot(shapes[0, :, 0], shapes[0, :, 1], 'o', label=name)[0])

    plt.title(title)
    plt.grid(False)
    if legend:
        plt.legend(
            fontsize=15, frameon=True, facecolor='w', edgecolor='r', framealpha=0.8
        )

    def update_line(input_tuple: tuple):
        index, frame = input_tuple
        image.set_data(frame)
        for line, (_, shapes) in zip(lines, named_shapes.items()):
            line.set_data(shapes[index, :, 0], shapes[index, :, 1])
        return (image, *lines)

    anim = animation.FuncAnimation(
        figure,
        update_line,
        enumerate(frames(video)),
        save_count=len(shapes),
        interval=interval,
        blit=True,
    )
    if save_to is not None:
        anim.save(save_to)
        plt.close()
    else:
        return anim2html(anim, html5)


class Visualizer(object):
    '''Visualizes videow with face landmarks either on live video or to file
    '''

    def __init__(self, fps=30, fourcc='VP80', color: tuple = (0, 255, 255)):
        '''
        Args:
            fps: frame fate of videos to visualize. This should be read from
                the video but sometimes it's corrupted, so this default is used
                Default for webcams is 30 for some reason.
            fourcc: type of codec to use. Typical choices:
                * 'XVID' - for .mp4 for fast encoding, lacks of formats support
                * 'MPEG' - for .mp4, .avi standard encoding
                * 'VP80' - Google's encoder, wide rage of fromats, a bit slow
                * 'VP90' - larger files, slower (maybe quality is better)
            color: 3 int tuple with RGB code of the color to draw points
        '''
        self.fps = fps
        self.fourcc = fourcc
        self.color = color

    def show(self, video: Path, shapes: np.ndarray, title: str = '') -> None:
        '''Plays video through cv2 windows
        Args:
            video: path to video file on disk
            shapes: result of :py:funct:`.extractors.VideoLandmarksExtractor.transform`
            title: window title
        '''
        delay = int(1000 / self.fps)

        for frame, shape in zip(frames(video), shapes):
            for point in shape:
                cv2.circle(frame, tuple(point), 1, self.color)
            cv2.imshow(title, frame)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def write(self, video: Path, shapes: Iterable[np.ndarray], out: Path):
        '''Writes video with shapes drawn on top
        '''
        with open_video(video) as in_video:
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            frame_shape = (
                int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            with open_video(out, 'w', fourcc, self.fps, frame_shape) as out_video:
                for frame, shape in zip(frames(in_video, rgb=False), shapes):
                    for point in shape:
                        cv2.circle(frame, tuple(point), 1, self.color)
                    out_video.write(frame)
