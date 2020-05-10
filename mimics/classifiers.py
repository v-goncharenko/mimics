import numpy as np
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from .transformers import Flattener, PointsToChannels


scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')

max_iter = 2500  # for Logistic Regression

# fmt: off
# from https://eeg-notebooks.readthedocs.io/en/latest/visual_p300.html
# and https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
clfs_full = {  # {model_name: (model, params_dict)}
    # 'pure LR': (
    #     make_pipeline(
    #         Flattener(),
    #         LogisticRegression(solver='saga', max_iter=max_iter),  # , l1_ratio=0.5),
    #     ),
    #     {
    #         'logisticregression__penalty': ('l1', 'l2'),  # , 'elasticnet'),
    #         'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
    #     },
    # ),

    # 'pure LDA': (
    #     make_pipeline(
    #         Flattener(),
    #         LDA(solver='lsqr', shrinkage='auto'),
    #     ),
    #     {},
    # ),

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
            'csp__n_components': (2, 3, 4),
            # 'csp__transform_into': ('average_power', 'csp_space'),
            'csp__log': (True, False),
            'csp__reg': ('empirical', 'shrunk', 'oas'),
            # 'csp__cov_est': ('concat', 'epoch'),
        },
    ),

    # from pyriemann.spatialfilters import Xdawn
    # 'Xdawn LDA': (
    #     make_pipeline(
    #         PointsToChannels(),
    #         Xdawn(),
    #         Flattener(),
    #         LDA(shrinkage='auto', solver='eigen'),
    #     ),
    #     {
    #         'xdawn__nfilter': (2, 4, 6),
    #         'xdawn__estimator': ('scm', 'lwf', 'oas'),
    #     },
    # ),

    'ERPCov TS LR': (
        make_pipeline(
            PointsToChannels(),
            ERPCovariances(estimator='oas'),
            TangentSpace(),
            LogisticRegression(solver='saga', max_iter=max_iter),  # , l1_ratio=0.5),
        ),
        {
            'erpcovariances__estimator': ('oas', 'lwf'),  # , 'scm'),
            'erpcovariances__svd': (None, 10),
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
            'erpcovariances__svd': (None, 10),
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

for name, (clf, _) in clfs_full.items():
    clf.name = name


clfs_small = {
    'pure LR': (
        make_pipeline(
            Flattener(),
            LogisticRegression(solver='saga', max_iter=max_iter),
        ),
        {
            'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
        },
    ),
}

for name, (clf, _) in clfs_small.items():
    clf.name = name

# fmt: off


__all__ = (
    'scores',
    'clfs_full',
    'clfs_small',
)
