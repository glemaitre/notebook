from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=100000, n_features=20, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=[0.01, 0.04, 0.95],
                           class_sep=0.8, random_state=0)

X_res, y_res = SMOTE().fit_sample(X, y)
