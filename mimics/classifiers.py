import numpy as np
from mne.decoding import CSP, Vectorizer
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .transformers import Flattener, PointsToChannels


scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')

# fmt: off
# from https://eeg-notebooks.readthedocs.io/en/latest/visual_p300.html
# and https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
clfs = {  # {model_name: (model, params_dict)}
    'pure LR': (
        make_pipeline(
            Flattener(),
            LogisticRegression(solver='saga', l1_ratio=0.5, max_iter=1000),
        ),
        {
            'logisticregression__penalty': ('l1', 'l2', 'elasticnet'),
            'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
        },
    ),

    'pure LDA': (
        make_pipeline(
            Flattener(),
            LDA(solver='lsqr', shrinkage='auto'),
        ),
        {},
    ),

    'pure SVM': (
        make_pipeline(
            Flattener(),
            SVC(),
        ),
        {
            'svc__C': np.exp(np.linspace(-4, 4, 3)),
            'svc__kernel': ('linear', 'rbf'),
        },
    ),

    'CSP LDA': (
        make_pipeline(
            PointsToChannels(),
            CSP(n_components=4),
            LDA(shrinkage='auto', solver='eigen'),
        ),
        {
            'csp__n_components': (2, 3, 4, 5, 7),
            'csp__transform_into': ('average_power', 'csp_space'),
            'csp__cov_est': ('concat', 'epoch'),
        },
    ),

    'Xdawn LDA': (
        make_pipeline(
            Xdawn(2, classes=[1]),
            Vectorizer(),
            LDA(shrinkage='auto', solver='eigen'),
        ),
        {
            'xdawn__nfilter': (2, 4, 6),
        },
    ),

    'ERPCov TS LR': (
        make_pipeline(
            PointsToChannels(),
            ERPCovariances(estimator='oas'),
            TangentSpace(),
            LogisticRegression(penalty='l2', solver='saga', l1_ratio=0.5, max_iter=1000),
        ),
        {
            'erpcovariances__estimator': ('oas', 'lwf', 'scm'),
            'logisticregression__penalty': ('l2', 'l1'),  # , 'elasticnet'),
            'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
        },
    ),

    'ERPCov MDM': (
        make_pipeline(
            PointsToChannels(),
            ERPCovariances(estimator='oas'),
            MDM(),
        ),
        {
            'erpcovariances__estimator': ('oas', 'lwf'),
        },
    ),

    'pure RF': (
        make_pipeline(
            Flattener(),
            RandomForestClassifier(n_estimators=100, max_depth=5),
        ),
        {
            'randomforestclassifier__max_depth': (3, 5, 10),
            'randomforestclassifier__n_estimators': (100, 200),
        },
    ),
}

for name, (clf, _) in clfs.items():
    clf.name = name
# fmt: off


__all__ = (
    'scores',
    'clfs',
)
