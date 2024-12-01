import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap

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
    plt.show()

#kpca = KernelPCA(n_components=9, kernel='rbf')
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca_train = kpca.fit_transform(X_train)
print("Kernel PCA X_train data:", X_kpca_train.shape)

# pca the test dataset
#X_kpca_test = pca.transform(X_test)
scatter_class(X_kpca_train, y_train)


# ISOMAP on PCA'd data, else ISOMAP take too long
#iso = Isomap(n_components=2, n_neighbors = 3)
#print('start isomap')
#X_iso_train = iso.fit_transform(X_pca_train)
#print('train done')
#X_iso_test = iso.transform(X_pca_test)
#scatter_class(X_iso_train, y_train)

####### Naive Guassian Bayes #######

gnb = GaussianNB()
gnb.fit(X_pca_train, y_train)
gnb_pred = gnb.predict(X_pca_test)
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, gnb_pred))

print(sum(y_test==1) / len(y_test))
print(sum(gnb_pred==1) / len(gnb_pred))

# ISOMAP testing


