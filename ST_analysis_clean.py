import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.manifold import Isomap
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import utils

path = '.\\data\\ST4000DX000_data.csv'
X, y = utils.import_smart_data('.\\data\\ST4000DX000_data.csv')
columns = X.columns

# SMOTE imbalanced data
sm = SMOTE(random_state=99, k_neighbors=1)
X_res, y_res = sm.fit_resample(X, y)

# shuffle and split train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=99)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# visualize column distribution
utils.plot_hist_by_class(X_train, y_train, 7,4 , columns)

# Gaussian Kernel PCA
#n_components = 9 # this was the number of components accounting for 95% of sum of eigenvalues
n_components = 2
kpca = KernelPCA(n_components=n_components, kernel='rbf')
X_kpca_train = kpca.fit_transform(X_train)
X_kpca_test = kpca.transform(X_test)
utils.plot_2d_pca(X_kpca_train, y_train)

# Gaussian Naive Bayes
gnb = GaussianNB() # default var_smoothing was fine
gnb.fit(X_kpca_train, y_train)
gnb_pred = gnb.predict(X_kpca_test)
utils.model_metrics(y_test, gnb_pred, "Gaussian Naive Bayes")

# Logistic Regression
logreg = LogisticRegression(random_state=99).fit(X_kpca_train, y_train)
logreg_pred = logreg.predict(X_kpca_test)
utils.model_metrics(y_test, logreg_pred, "Logistic Regression")

# KNN
knn = KNeighborsClassifier(n_neighbors=4) # N=4 was one of best performers
knn.fit(X_kpca_train, y_train)
knn_pred = knn.predict(X_kpca_test)
utils.model_metrics(y_test, knn_pred, "KNN")

# plot classifier comparison
classifiers = [gnb, logreg, knn]
names = ["Gaussian NB", "Logistic Regression", "KNN N=4"]
utils.plot_classifer_comparison(classifiers, names, X_kpca_train, y_train)