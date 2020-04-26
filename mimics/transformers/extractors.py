import warnings
from pathlib import Path
from typing import Iterable, List

import dlib
import face_alignment
import numpy as np
import SAN
import torch
from joblib import Parallel, delayed

from ..types import File, Optional
from ..utils import frames
from .basic import Transformer


# groups of indexes according to 300-W dataset https://ibug.doc.ic.ac.uk/resources/300-W/
inds_68 = {
    'nose': tuple(range(27, 36)),
    'right_brow': tuple(range(17, 22)),
    'left_brow': tuple(range(22, 27)),
    'brows': tuple(range(17, 27)),
    'lips': tuple(range(48, 68)),
    'right_eye': tuple(range(36, 42)),
    'left_eye': tuple(range(42, 48)),
    'eyes': tuple(range(36, 48)),
}


class VideoLandmarksExtractor(Transformer):
    '''Extracts face landmarks from given videos

    Provides scarfold implementations and signatures for methods
    '''

    def transform(self, videos: Iterable[Path]) -> List[np.ndarray]:
        '''Extracts face landmarks from given videos

        Args:
            videos: iterable of paths to videos to extract points from

        Returns:
            list with ndarrays corresponding to each of given videos
                each array shaped (#frames, 68, 2) - for each frame in video
                68 points of dlib extractor with x and y coordinates in the last dimention

        Retruning datatype is list cause videos may have different frames count
        You may whsh to np.stack results in case of equal length.
        '''
        return self._transform(videos)


class DlibExtractor(VideoLandmarksExtractor):
    '''Extraction based on dlib trained models

    Pretrained models taken from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    '''

    default_predictor_path = (
        Path(__file__).resolve().parent.parent
        / 'models/dlib/shape_predictor_68_face_landmarks.dat'
    )

    indexes = inds_68

    def __init__(self, predictor_path: Optional[Path] = None, n_jobs: int = -1):
        '''
        Args:
            predictor_path: path to file `shape_predictor_68_face_landmarks.dat`
                which could be downloaded from oficial dlib site
            n_jobs: parameter for joblib
        '''
        self.predictor_path = predictor_path or DlibExtractor.default_predictor_path
        self.n_jobs = n_jobs

    def _transform(self, videos):
        '''Implements :py:funct:`.VideoLandmarksExtractor.transform`
        '''
        delayed_extract = delayed(DlibExtractor.extract_shapes)
        dataset = Parallel(n_jobs=self.n_jobs)(
            delayed_extract(video_path, self.predictor_path) for video_path in videos
        )
        return dataset

    @staticmethod
    def detect_face(image):
        '''Detects face on image using dlib

        Returns None if no face appears on image,
            raises ValueError if multiple faces detected
        '''
        detector = dlib.get_frontal_face_detector()
        faces = detector(image)
        if len(faces) > 1:
            raise ValueError(f'More than one face on the video! Have {len(faces)}.')
        if len(faces) == 0:
            return None
        return faces[0]

    @staticmethod
    def extract_shapes(
        video_path: Path, predictor_path: Optional[Path] = None
    ) -> np.ndarray:
        '''Extracts points from given video

        Note: this function assumes that person's face doesn't change position
            on all frames!

        Returns:
            ndarray shaped (#frames, 68, 2) - face landmarks for each frame
        '''
        predictor_path = predictor_path or DlibExtractor.default_predictor_path

        # first detect face position
        for frame in frames(video_path):
            face = DlibExtractor.detect_face(frame)
            if face is not None:
                break

        predictor = dlib.shape_predictor(predictor_path.as_posix())
        extracted = []

        for frame in frames(video_path):
            shape = predictor(frame, face)
            points = np.array([(s.x, s.y) for s in shape.parts()])
            extracted.append(points)
        return np.stack(extracted)


class FaExtractor(VideoLandmarksExtractor):
    '''Face-Alignment based extractor https://github.com/1adrianb/face-alignment
    '''

    def __init__(self, device=None, *, verbose: bool = False):
        self.device = str(device or 'cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cpu':
            warnings.warn(
                'Using CPU calculations which will take a loooong time to evaluate'
            )

        self.extractor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device=self.device
        )
        self.verbose = verbose

    def _transform(self, videos):
        dataset = []
        for i, video_path in enumerate(videos):
            if self.verbose:
                print(f'Extracting {i} of {len(videos)}: {video_path.name}')

            # first detect face position
            for frame in frames(video_path):
                faces = self.extractor.face_detector.detect_from_image(frame)
                if len(faces) == 1:
                    break

            shapes = np.array(
                [
                    self.extractor.get_landmarks(frame, faces)[0]
                    for frame in frames(video_path)
                ]
            )
            dataset.append(shapes)
        return dataset


class SanExtractor(VideoLandmarksExtractor):
    '''Source https://github.com/v-goncharenko/landmark-detection

    Pretrained model file https://drive.google.com/file/d/18YV8RxuTny6WrWc1eI2B4g3rM_NxtjEi
    '''

    def __init__(
        self,
        model_path: File = '../models/landmark_detection/checkpoint_49.pth.tar',
        device=None,
        *,
        verbose: bool = False,
    ):
        self.detector = SAN.SanLandmarkDetector(model_path, device)
        self.verbose = verbose

    def _transform(self, videos):
        dataset = []
        for i, video_path in enumerate(videos):
            if self.verbose:
                print(f'Extracting {i} of {len(videos)}: {video_path.name}')

            # first detect face position via Dlib
            for frame in frames(video_path):
                face = DlibExtractor.detect_face(frame)
                if face is not None:
                    break
            face = (face.left(), face.top(), face.right(), face.bottom())

            shapes = np.array(
                [self.detector.detect(frame, face)[0] for frame in frames(video_path)]
            )
            dataset.append(shapes)
        return dataset
