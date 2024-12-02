from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import utils

#model = 'WDC WD1600AAJS'
#model= 'WDC WD20EFRX'
model = 'ST4000DX000'
path = '.\\data\\' + model + '_data.csv'
X, y = utils.import_smart_data(path)
columns = X.columns

# SMOTE imbalanced data
# k_neighbors=1 if WDC WDC WD1600AAJS else error
# k_neighbors=2 if WDC WD20EFRX else error
# k_neighbors=1 if ST4000DX000 else error
sm = SMOTE(random_state=99, k_neighbors=1)
X_res, y_res = sm.fit_resample(X, y)

# shuffle and split train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=99)

# standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# visualize column distribution
utils.plot_hist_by_class(X_train, y_train, 6,3 , columns)

# Gaussian Kernel PCA
n_components = utils.get_best_kpca_n_components(X_train)
#n_components = 2
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
n_neighbors = utils.get_best_knn_neighbors(X_kpca_train, y_train)
knn = KNeighborsClassifier(n_neighbors=n_neighbors) # N=4 was one of best performers
knn.fit(X_kpca_train, y_train)
knn_pred = knn.predict(X_kpca_test)
utils.model_metrics(y_test, knn_pred, "KNN")

# plot model boundary for train data
classifiers = [gnb, logreg, knn]
names = ["Gaussian NB", "Logistic Regression", "KNN"]
utils.plot_classifer_comparison(classifiers, names, X_kpca_train, y_train)

#plot model boundary for test data
preds = [gnb_pred, logreg_pred, knn_pred]
utils.plot_classifer_comparison(classifiers, names, X_kpca_test, preds, y_is_pred=True)

'''
################### Try predicting on completely different disk model #######################
# difficult - different models from same manufacturers have diff smart statistics populated
# forcing one model to keep the same statistics as another means meaningful data may be dropped
model = 'TOSHIBA MD04ABA400V'
path = '.\\data\\' + model + '_data.csv'

data = pd.read_csv(path)
data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
data.drop(['Unnamed: 0', 'date', 'model', 'serial_number'], inplace=True, axis=1)
print(data.groupby('failure').size())
y2 = data['failure']
X2 = data.drop(['failure'], axis=1).fillna(0)  # replace NAs with 0
drop_col = []
for col in X2.columns:
    if col not in columns: # columns is from fitted model's data
        drop_col.append(col)
X2 = X2.drop(drop_col, axis=1)
X2 = scaler.transform(X2)

utils.plot_hist_by_class(X2, y2, 7,4 , columns)
X2_kpca = kpca.transform(X2)

gnb_pred_2 = gnb.predict(X2_kpca)
#utils.model_metrics(y2, gnb_pred_2, "Gaussian NB")

logreg_pred_2 = logreg.predict(X2_kpca)
#utils.model_metrics(y2, logreg_pred_2, "Logistic Regression")

knn_pred_2 = knn.predict(X2_kpca)
#utils.model_metrics(y2, knn_pred_2, "KNN")

predictions = [gnb_pred_2, logreg_pred_2, knn_pred_2]
utils.plot_classifer_comparison(classifiers, names, X2_kpca, predictions, y_is_pred=True)
'''






