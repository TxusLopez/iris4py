# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:05:50 2015

@author: Txus López
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from itertools import cycle
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn import tree
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.plot(data[target == i, 0],data[target == i, 1], 'o',c=c, label=label)
    pl.legend(target_names)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# import some data to play with
iris = datasets.load_iris()

print(iris.data.shape) #150 instances and 4 attributes
print(np.unique(iris.target))

X, y = iris.data, iris.target
#plot_2D(X, y, iris.target_names)

#----------------------------------------------------------------------
# First figure: PCA
#pca = PCA(n_components=2, whiten=True).fit(X)
#X_pca = pca.transform(X)
#plot_2D(X_pca, y, iris.target_names)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)  
cv = ShuffleSplit(X_train.shape[0], n_iter=50, test_size=0.2, random_state=0)


############ TREE ##########################################################################################
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X_train, y_train)

print("TREE Score: ")
print(clf1.score(X_test, y_test)) 

print("TREE  Classification report")
y_pred1 = clf1.predict(X_test)

cr1=classification_report(y_test, y_pred1)
print(cr1)

print("TREE Confusion matrix")
cm1=confusion_matrix(y_test, y_pred1, labels=range(3))
print(cm1)  

plt.matshow(cm1)
plt.title('TREE Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'TREE Learning Curves'
plot_learning_curve(clf1, title_string, X_train, y_train, cv=cv)
plt.show()

############ SVM ##########################################################################################
param_grid = {'C': [1, 10, 100, 1000],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf2 = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid, cv=cv)
clf2.fit(X_train, y_train)
clf2 = clf2.best_estimator_

print("SVM Score: ")
print(clf2.score(X_test, y_test)) 

print("SVM  Classification report")
y_pred2 = clf2.predict(X_test)

cr2=classification_report(y_test, y_pred2)
print(cr2)

print("SVM Confusion matrix")
cm2=confusion_matrix(y_test, y_pred2, labels=range(3))
print(cm2)  
        
plt.matshow(cm2)
plt.title('SVM Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'SVM Learning Curves'
plot_learning_curve(clf2, title_string, X_train, y_train, cv=cv)
plt.show()

############ Naive Bayes ##########################################################################################
gnb = GaussianNB()
clf3 = gnb.fit(X_train, y_train)

print("NB Score: ")
print(clf3.score(X_test, y_test)) 

print("NB  Classification report")
y_pred3 = clf3.predict(X_test)

cr3=classification_report(y_test, y_pred3)
print(cr3)

print("NB Confusion matrix")
cm3=confusion_matrix(y_test, y_pred3, labels=range(3))
print(cm3)  
        
plt.matshow(cm3)
plt.title('NB Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'NB Learning Curves'
plot_learning_curve(clf3, title_string, X_train, y_train, cv=cv)
plt.show()

############ Logistic Regression ##########################################################################################
clf4 = LogisticRegression()
clf4.fit(X_train, y_train)

print("LR Score: ")
print(clf4.score(X_test, y_test)) 

print("LR  Classification report")
y_pred4 = clf4.predict(X_test)

cr4=classification_report(y_test, y_pred4)
print(cr4)

print("LR Confusion matrix")
cm4=confusion_matrix(y_test, y_pred4, labels=range(3))
print(cm4)  
        
plt.matshow(cm4)
plt.title('LR Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'LR Learning Curves'
plot_learning_curve(clf4, title_string, X_train, y_train, cv=cv)
plt.show()

############ Random Forest ##########################################################################################
clf5 = RandomForestClassifier(n_estimators=10)
clf5.fit(X_train, y_train)

print("RF Score: ")
print(clf5.score(X_test, y_test)) 

print("RF  Classification report")
y_pred5 = clf5.predict(X_test)

cr5=classification_report(y_test, y_pred5)
print(cr5)

print("RF Confusion matrix")
cm5=confusion_matrix(y_test, y_pred5, labels=range(3))
print(cm5)  
        
plt.matshow(cm5)
plt.title('RF Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'RF Learning Curves'
plot_learning_curve(clf5, title_string, X_train, y_train, cv=cv)
plt.show()

############ Ada Boost ##########################################################################################
clf6 = AdaBoostClassifier(n_estimators=100)
clf6.fit(X_train, y_train)

print("AB Score: ")
print(clf6.score(X_test, y_test)) 

print("AB  Classification report")
y_pred6 = clf6.predict(X_test)

cr6=classification_report(y_test, y_pred6)
print(cr6)

print("AB Confusion matrix")
cm6=confusion_matrix(y_test, y_pred6, labels=range(3))
print(cm6)  
        
plt.matshow(cm6)
plt.title('AB Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show() 

# Learning curves
title_string = 'AB Learning Curves'
plot_learning_curve(clf6, title_string, X_train, y_train, cv=cv)
plt.show()

######################################################################################################

# Persistence model
#joblib.dump(clf1, 'tree_model_Irisdataset.pkl') 
#joblib.dump(clf2, 'svm_model_Irisdataset.pkl') 
#joblib.dump(clf3, 'nb_model_Irisdataset.pkl') 
#joblib.dump(clf4, 'lr_model_Irisdataset.pkl') 
