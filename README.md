# Detecting mimics disorders by face exercises videos

Задача состоит в отделении людей с нарушениями мимики от условно здоровых по видеозаписям мимических упражнений

## Initial configuration

`setup.py` is planned, for now manual installation

For usage:

1. install miniconda (or `virtualenv` although not tested)
2. create `mimics` env (see `.python-version` for version)
3. install building tools: `sudo apt update && sudo apt install -y build-essential`
4. `cmake`: `sudo snap install cmake --classic` works for ubuntu (this is needed for `dlib`, [instructions for macos](https://stackoverflow.com/questions/54719496/installing-dlib-in-python-on-mac/63140828))
5. seems like this is needed for OpenCV to work on Ubuntu: `sudo apt install -y libsm6 libxext6 libxrender-dev`
6. install `ffmpeg`: `sudo apt install -y ffmpeg`
7. load `dlib` weights with `mimics/load_weights.sh` (run it from its parent directory)
8. For usage just `pip install -r requirements.txt`

For development:

* install `dvc`
* install `pre-commit` (I prefer to install it separately)
* run `pip install -r requirements-dev.txt`
* run `pre-commint install`
* run `dvc pull` to get all datasets and models (may take a few dozens of minutes)

## Experiments reproduction

Available through `commands.py` where each function exposes and experiment.
Experiment either logs results to `mlflow` or writes artifact files to `data/tmp`

## Optimal params (alpha)

* brows (alpha)
  * low: 0.45, high: 5.0
* smile (alpha)
  * low: 0.65, high: 7.0

# General scheme

1. Feature extraction: для каждого кадра видео выделение положения лица (предобученные сетки)
2. Preprocessing: сглаживание во времени, центрирование, проективная нормализация
3. Classification and analysis: обучение статистической модели на подготовленных данных, а также выделение высокоуровневых семантичных фичей, которые могут быть проанализированы врачом визуально

## Feature extraction

Существует множество вариантов выделения очертаний лица. В секции Sources рассмотрены основные списки, далее вкрадце описаны наиболее доступные и современные рассмотрены методы.

Очертания лица(shape) как правило определяются для каждого кадра отдельно (единицы работ посвящённх именно видео см. Supervision-by-Registration), что влечёт за собой особенности типа случайного шума на последовательных кадрах.

Эти способы реализованы в виде sklearn-style трансформеров в файле `extractors.py`

У некоторых детекторов есть 2 режима работы: определение 68 точек, либо 21 точки. Это вызвано тем, что есть 2 типа разметки датасетов: 300W-like и AFLW-like.

Points are reffered "left" and "right" always from person's point of view.

### Sources, lists

[Обширный лист решений для детекции очертаний](https://github.com/mrgloom/Face-landmarks-detection-benchmark). Он несколько не структурирован, местами ссыли устарели, но можно надеяться на большой recall. Кажется, он даже дополняется и как-то поддерживается. Тут же есть пачка датасетов.

Также хорошим источником является [paperswithcode](https://paperswithcode.com/task/facial-landmark-detection), там указаны СОТЫ. Этот раздел ещё не проанализирован полностью https://paperswithcode.com/task/face-alignment

### Dlib

Используется библиотека [dlib](https://github.com/davisking/dlib)
Порядок действий:

* производится определение положения лица функцией `get_frontal_face_detector`
* далее на основе этого функцией `shape_predictor` и параметров [`shape_predictor_68_face_landmarks`](https://github.com/davisking/dlib-models) выделяется [68 точек](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/), являющихся очертаниями лица человека

*Плюсы*:

* работает локально на CPU
* легодоступна, интегрирована в нашу систему
* стабильна (относительно программной совместимости)

*Минусы*:
* очень слабо работает с движениями бровей/губ, сильно шумит при мимических движениях, постоянно сбивается

### Face Alignment

https://github.com/1adrianb/face-alignment

### SAN

https://arxiv.org/pdf/1803.04108v4.pdf

### OpenCV

Вроде бы имеет несколько методов определения очертаний (AAM, LBF), но пишут, что под них сложно найти предобученные сети.

[link](https://github.com/kurnianggoro/GSOC2017) - ссылки на статьи, где описаны методы, вроде бы даже есть предобученные модели, всё на плюсах

### HRNets

[link](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) -

### Face Recognition

https://github.com/ageitgey/face_recognition

### OpenPose

Very hard to install and get to work (on Ubuntu)

https://github.com/CMU-Perceptual-Computing-Lab/openpose

### Supervision-by-Registration

https://github.com/facebookresearch/supervision-by-registration

### InsightFace

https://github.com/deepinsight/insightface#third-party-re-implementation

### AdaptiveWingLoss

https://github.com/protossw512/AdaptiveWingLoss

### Adaloss

https://arxiv.org/pdf/1908.01070v1.pdf
Многообещающая статья, но без реализации

### Action Unit Detection

Технология определения активных частей лица на видео. не выделяет точек непосредственно, но может быть использованна косвенно для извлечения фичей/активаций.
Использует optical flow для анализа видео, можно позаимстовать реализацию.

Dataet+Challenge - https://arxiv.org/pdf/1702.04174.pdf

AUNets - https://arxiv.org/pdf/1704.07863v2.pdf
Использует optical flow видео для предсказаний. Очень громоздкая. Не увидел pretrained

AU R-CNN - https://arxiv.org/pdf/1812.05788v2.pdf
Эта сетка может предсказывать активации регионов лица по _одной_ фотографии, То есть по каждому кадру. Таким образом можно получить временной ряд активаий интересующего нас региона или по крайней мере эмбэддинги таких активаций (можно барть активации предпоследнего слоя перед классификационным). Может быть получится такие цифры легче и точнее агрегировать.

## Preprocessing

В файле `transformers.py` содержатся все имеющиеся варианты трансформеров.

Текущая оптимальная схема выдаётся в функции `get_preprocessing`

Ideas:

* augment time series (flip, multiplication, etc) as well as videos before extracting points

## Classification and analysis

### Methods for feature extraction from brows oscilation

* statsmodels.tsa.seasonal.seasonal_decompose - apply to mean of all channels
    plot detected frequency and try to split by it
* scipy findpeaks -
* fft analysis

### Classification algos variants

* лин. модели как бейзлайн
* леса над семантическими фичами
* сетки: автоэнкодер + FC predictor
