from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lcf = LazyClassifier()
models, predictions = lcf.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print(models.to_markdown())

""" RESULTS:

| Model                         |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|:------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
| LogisticRegression            |   0.976608 |            0.972511 |  0.972511 |   0.976539 |   0.0315015  |
| SGDClassifier                 |   0.97076  |            0.970563 |  0.970563 |   0.9708   |   0.00999999 |
| LinearSVC                     |   0.97076  |            0.967749 |  0.967749 |   0.970718 |   0.0180001  |
| ExtraTreesClassifier          |   0.97076  |            0.964935 |  0.964935 |   0.970626 |   0.127992   |
| SVC                           |   0.97076  |            0.964935 |  0.964935 |   0.970626 |   0.0160041  |
| RandomForestClassifier        |   0.97076  |            0.964935 |  0.964935 |   0.970626 |   0.195027   |
| LGBMClassifier                |   0.97076  |            0.964935 |  0.964935 |   0.970626 |   0.0949969  |
| BernoulliNB                   |   0.959064 |            0.961039 |  0.961039 |   0.959223 |   0.00999761 |
| KNeighborsClassifier          |   0.964912 |            0.957359 |  0.957359 |   0.964692 |   0.0160048  |
| PassiveAggressiveClassifier   |   0.964912 |            0.954545 |  0.954545 |   0.964564 |   0.0109999  |
| RidgeClassifier               |   0.964912 |            0.954545 |  0.954545 |   0.964564 |   0.0119734  |
| CalibratedClassifierCV        |   0.964912 |            0.954545 |  0.954545 |   0.964564 |   0.04       |
| Perceptron                    |   0.953216 |            0.953463 |  0.953463 |   0.953341 |   0.0119731  |
| BaggingClassifier             |   0.959064 |            0.952597 |  0.952597 |   0.958877 |   0.0499935  |
| QuadraticDiscriminantAnalysis |   0.959064 |            0.952597 |  0.952597 |   0.958877 |   0.0109999  |
| XGBClassifier                 |   0.953216 |            0.950649 |  0.950649 |   0.953216 |   0.0939975  |
| RidgeClassifierCV             |   0.959064 |            0.94697  |  0.94697  |   0.958578 |   0.0139959  |
| LinearDiscriminantAnalysis    |   0.953216 |            0.939394 |  0.939394 |   0.952566 |   0.0270059  |
| GaussianNB                    |   0.94152  |            0.935498 |  0.935498 |   0.941346 |   0.0100007  |
| NuSVC                         |   0.947368 |            0.934632 |  0.934632 |   0.946744 |   0.0220513  |
| AdaBoostClassifier            |   0.94152  |            0.932684 |  0.932684 |   0.941153 |   0.135037   |
| LabelSpreading                |   0.94152  |            0.932684 |  0.932684 |   0.941153 |   0.0249672  |
| LabelPropagation              |   0.94152  |            0.932684 |  0.932684 |   0.941153 |   0.020997   |
| NearestCentroid               |   0.923977 |            0.912771 |  0.912771 |   0.923364 |   0.00952387 |
| DecisionTreeClassifier        |   0.906433 |            0.892857 |  0.892857 |   0.905505 |   0.0150049  |
| ExtraTreeClassifier           |   0.900585 |            0.890909 |  0.890909 |   0.900129 |   0.0100021  |
| DummyClassifier               |   0.614035 |            0.5      |  0.5      |   0.467201 |   0.0090003  |
"""
