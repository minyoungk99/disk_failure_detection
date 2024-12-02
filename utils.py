import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap
import numpy as np

'''
Reads in single model's data and drops unnecessary columns
Returns data in (X, y) tuple, where y is the failure column
'''
def import_smart_data(path):
    data = pd.read_csv(path)
    data.drop(list(data.filter(regex="raw$")), axis=1, inplace=True)
    data.drop(['Unnamed: 0', 'date', 'model', 'serial_number'], inplace=True, axis=1)
    y = data['failure']
    X = data.drop(['failure'], axis=1).fillna(0) # replace NAs with 0
    drop_col = []
    for col in X.columns:
        a = X[col]
        if len(a) == (sum(a==0)):
            drop_col.append(col)
    X = X.drop(drop_col, axis=1)

    return (X, y)


'''
Plots a histogram of all columns in X. Data associated wity y=1 (failure) in red, y=0 (working) in blue
X must be a pandas dataframe.
'''
def plot_hist_by_class(X, y, nrow, ncol):
    working = X[y == 0]
    failure = X[y == 1]
    columns = X.columns
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

'''
Scatter plot of data that has been reduced to 2D via PCA.
Points in blue are y=0, working drives.
Points in red are y=1, failed drives.
'''
def plot_2d_pca(X, y):
    good = X[y == 0, :]
    fail = X[y == 1, :]
    plt.scatter(good[:, 0], good[:, 1], color='b', marker='.')
    plt.scatter(fail[:, 0], fail[:, 1], color='r', marker='.')
    plt.show()

'''
Given a y_true and y_pred from classifier model, plots confusion matrix and prints ROC AUC score.
'''
def model_metrics(y_true, y_pred, model_name=''):
    # plot confusion matrix
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)
    plt.title(model_name + " Confusion Matrix")
    plt.show()

    # ROC AUC
    print(model_name + " ROC AUC score:", round(roc_auc_score(y_true, y_pred), 3))


'''
Code referenced from OMSA ISYE6740 Module 9 demo code
Given a list of fitted classifiers,  list of classifer names in string, X and y data,
plots the data colored by y and the classifier's decision boundaries.
'''
def plot_classifer_comparison(classifiers, names, X, y):
    # 2d plots only
    if(X.shape[1] == 2):
        # define meshgrid
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        h = 0.2
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
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
            # ax.scatter(X_kpca_train[:, 0], X_kpca_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', marker='.', alpha=0.5)
            # Plot the testing points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
        plt.tight_layout()
        plt.show()