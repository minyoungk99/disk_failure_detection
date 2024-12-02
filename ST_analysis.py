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

'''
def plot_by_date(X, nrow, ncol):
    fig, axs = plt.subplots(nrow, ncol)
    X.sort_values('date', inplace=True)
    for i in range(nrow):
        for j in range(ncol):
            if (ncol * i + j) < X.shape[1]:
                if X.columns[ncol*i+j] not in ['Unnamed: 0', 'date', 'model']:
                    feature = X.iloc[:, ncol*i+j]
                    axs[i, j].plot(feature)
                    axs[i, j].set_xticklabels([])
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_title(X.columns[ncol * i + j], fontsize=8)

    plt.show()
'''

# ST6000DX000
data = pd.read_csv('.\\data\\ST4000DX000_data.csv')
#data = pd.read_csv('.\\data\\ST4000DM000_data.csv')
data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)

# pick serial number  Z305B2QN, sort by date, and see if any stats are cumulative
# SMART 7, 9, 193 look cumulative? but we can't tell for sure - maybe it's just increasing value like "total cycles"
# not enough information to say it's cumulative. write as next steps in paper - more info
'''
cumul = data[data['serial_number'] ==  'Z305B2QN']
cumul.sort_values('date', inplace=True)
for i in range(3,cumul.shape[1]):
    plt.plot(cumul.iloc[:, i])
    plt.title(cumul.columns[i])
    plt.show()
'''

# dont need model, serial_number, date. also can't smote string col
data.drop(['Unnamed: 0', 'date', 'model', 'serial_number'], inplace=True, axis=1)
y = data['failure']
X = data.drop(['failure'], axis=1).fillna(0) # replace NAs with 0
drop_col = []
for col in X.columns:
    a = X[col]
    if len(a) == (sum(a==0)):
        drop_col.append(col)
X = X.drop(drop_col, axis=1)

# print(data.groupby('failure').size())
# 1.681M ok, 139 failures. Data is very imbalanced, so we will use SMOTE to make 50/50 classes
sm = SMOTE(random_state=99, k_neighbors=1)
X_res, y_res = sm.fit_resample(X, y)

# shuffle, split train/test
# train 2522001 rows, test 840667 rows
columns = X_res.columns
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=99)

# train standardScaler on train data and transform X_train, X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# plot distribution of working disks in blue, failed disks in red for all columns
# SMART 9 looks pretty important in differentiating classes

def plot_hist_by_class(working, failure, nrow, ncol):
    fig, axs = plt.subplots(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            if (ncol*i+j) < working.shape[1]:
                axs[i, j].hist(working[:, ncol*i + j], alpha=0.5, color='b', bins=15)
                axs[i, j].hist(failure[:, ncol*i + j], alpha=0.5, color='r', bins=15)
                axs[i,j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_title(columns[ncol*i + j], fontsize=8, pad=0)
    plt.show()

working = X_train[y_train == 0]
failure = X_train[y_train == 1]
#plot_hist_by_class(working, failure, 7, 7)


####### PCA #######
# plot from https://www.baeldung.com/cs/pca
#pca_test = PCA().fit(X_train)
#plt.plot(np.cumsum(pca_test.explained_variance_ratio_), label="Cumulative Explained Variance")
#plt.xlabel('Number of Components')
#plt.ylabel('Cumulative Explained Variance')
#plt.hlines(0.95, 0, 45, alpha=0.5, color='r', label="95% Explained Variance")
#plt.legend()
#plt.show()

# first 11 components explain 94.5% of variance. First 12 are 97.4%
#for ind, val in enumerate(np.cumsum(pca.explained_variance_ratio_)):
#    print(ind, val)

# so do another PCA with 11 components to perform dimensionality reduction
def scatter_class(X, y):
    good = X[y==0,:]
    fail = X[y==1,:]
    plt.scatter(good[:,0], good[:,1], color='b', marker='.')
    plt.scatter(fail[:,0], fail[:,1], color='r', marker='.')
    plt.show()

#pca = PCA(n_components=11)
#pca = PCA(n_components=2)
#X_pca_train = pca.fit_transform(X_train)
#print("PCA X_train data:", X_pca_train.shape)

# pca the test dataset
#X_pca_test = pca.transform(X_test)
#scatter_class(X_pca_train, y_train)

####### Kernel PCA #######
# plot from https://www.baeldung.com/cs/pca
#kpca_test = KernelPCA(kernel='rbf', n_components=30).fit(X_train)
#total_eig = sum(kpca_test.eigenvalues_)
#plt.plot(np.cumsum(kpca_test.eigenvalues_), label="Cumulative Sum of Eigenvalues")
#plt.xlabel('Number of Components')
#plt.ylabel('Cumulative Sum of Eigenvalues')
#plt.hlines(0.95*total_eig, 0, 45, alpha=0.5, color='r', label="95% Explained Variance")
#plt.legend()
#plt.show()

# first 9 componnents are 95% of eigenvalues
#for ind, val in enumerate(np.cumsum(kpca_test.eigenvalues_)):
#    print(ind, val)

# so do another PCA with 9 components to perform dimensionality reduction
def scatter_class(X, y):
    good = X[y==0,:]
    fail = X[y==1,:]
    plt.scatter(good[:,0], good[:,1], color='b', marker='.')
    plt.scatter(fail[:,0], fail[:,1], color='r', marker='.')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    plt.scatter(good[:, 0], good[:, 2], color='b', marker='.')
    plt.scatter(fail[:, 0], fail[:, 2], color='r', marker='.')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 3")
    plt.show()
    plt.scatter(good[:, 0], good[:, 3], color='b', marker='.')
    plt.scatter(fail[:, 0], fail[:, 3], color='r', marker='.')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 4")
    plt.show()

n_components = 2
kpca = KernelPCA(n_components=n_components, kernel='rbf')
X_kpca_train = kpca.fit_transform(X_train)
#print("Kernel PCA X_train data:", X_kpca_train.shape)

# pca the test dataset
X_kpca_test = kpca.transform(X_test)
#scatter_class(X_kpca_train, y_train)


# ISOMAP on PCA'd data, else ISOMAP take too long
#iso = Isomap(n_components=2, n_neighbors = 3)
#print('start isomap')
#X_iso_train = iso.fit_transform(X_pca_train)
#print('train done')
#X_iso_test = iso.transform(X_pca_test)
#scatter_class(X_iso_train, y_train)

def model_metrics(y_true, y_pred, model_name=''):
    # plot confusion matrix
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)
    plt.title(model_name + " Confusion Matrix")
    plt.show()

    # ROC AUC
    print(model_name + " ROC AUC score:", round(roc_auc_score(y_true, y_pred),3))

####### Naive Guassian Bayes #######
# default var_smoothing seems fine
'''
cv_scores = []
smoothings = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
for i in smoothings:
    print(i)
    gnb_test = GaussianNB(var_smoothing=i)
    gnb_test.fit(X_kpca_train, y_train)
    cv_results = cross_validate(gnb_test, X_kpca_train, y_train, cv=5)
    print(cv_results['test_score'])
    cv_scores.append(np.mean(cv_results['test_score']))

plt.plot(cv_scores)
plt.xlabel("var_smoothing")
plt.ylabel("5-fold CV Mean Score")
plt.title("5-fold CV Mean Score vs var_smoothing for Gaussian NB")
plt.show()

print(cv_scores)
'''

gnb = GaussianNB()
gnb.fit(X_kpca_train, y_train)
gnb_pred = gnb.predict(X_kpca_test)
model_metrics(y_test, gnb_pred, "Gaussian Naive Bayes")

####### Logistic Regression #######
logreg = LogisticRegression(random_state=99).fit(X_kpca_train, y_train)
logreg_pred = logreg.predict(X_kpca_test)
model_metrics(y_test, logreg_pred, "Logistic Regression")

####### KNN #######
# based on below commented code, pick K=4
'''
cv_scores = []
n_neighbors = np.arange(1, 20)
for i in n_neighbors:
    knn_test = KNeighborsClassifier(n_neighbors=i)
    knn_test.fit(X_kpca_train, y_train)
    cv_results = cross_validate(knn_test, X_kpca_train, y_train, cv=5)
    print(cv_results['test_score'])
    cv_scores.append(np.mean(cv_results['test_score']))

plt.plot(n_neighbors,cv_scores)
plt.xlabel("Number of Neighbors")
plt.ylabel("5-fold CV Mean Score")
plt.title("5-fold CV Mean Score vs # of Neighbors for KNN")
plt.show()

print(n_neighbors[np.max(cv_scores) == cv_scores])
print(cv_scores)
'''

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_kpca_train, y_train)
knn_pred = knn.predict(X_kpca_test)
model_metrics(y_test, knn_pred, "KNN")

#visualize decision boundaries
# referenced plot_classifier_comparison.py from module 9 demo code
if n_components == 2:
    classifiers = [gnb, logreg, knn]
    names = ["Gaussian NB", "Logistic Regression", "KNN N=4"]

    # define meshgrid
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    h = 0.2
    x_min, x_max = X_kpca_train[:, 0].min() - .5, X_kpca_train[:, 0].max() + .5
    y_min, y_max = X_kpca_train[:, 1].min() - .5, X_kpca_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    figure = plt.figure(figsize=(13, 5))
    for i in range(len(classifiers)):
        name = names[i]
        clf = classifiers[i]

        ax = plt.subplot(1, len(classifiers), i + 1)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points - uncomment to plot training data points
        ax.scatter(X_kpca_train[:, 0], X_kpca_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', marker='.', alpha=0.5)
        # Plot the testing points
        #ax.scatter(X_kpca_test[:, 0], X_kpca_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #        size=15, horizontalalignment='right')

    plt.tight_layout()
    plt.show()




