from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import utils

import numpy as np

log_reg = LogisticRegression()
naive_bayes = GaussianNB()
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier()
svm = SVC()

print 'reading data...'
features, labels = utils.read_from_csv(one_hot = False, filename = './data/fashion-mnist_train.csv')

print 'fitting...'
log_reg.fit(features[:-100], labels[:-100])
print(log_reg)
log_reg_pred = log_reg.predict(features[-100:])
print 'Logistic Regression Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(labels[-100:], log_reg_pred))
print(metrics.confusion_matrix(labels[-100:], log_reg_pred))


naive_bayes.fit(features[:-100], labels[:-100])
print(naive_bayes)
nb_pred = naive_bayes.predict(features[-100:])
print 'Naive Bayes Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(labels[-100:], nb_pred))
print(metrics.confusion_matrix(labels[-100:], nb_pred))


knn.fit(features[:-100], labels[:-100])
print(knn)
knn_pred = knn.predict(features[-100:])
print 'K-Nearest Neighbors Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(labels[-100:], knn_pred))
print(metrics.confusion_matrix(labels[-100:], knn_pred))


decision_tree.fit(features[:-100], labels[:-100])
print(decision_tree)
dec_tree_pred = decision_tree.predict(features[-100:])
print 'Decision Tree Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(labels[-100:], dec_tree_pred))
print(metrics.confusion_matrix(labels[-100:], dec_tree_pred))


svm.fit(features[:-100], labels[:-100])
print(svm)
svm_pred = svm.predict(features[-100:])
print 'Support Vector Machine Results:'
print '- - - - - - - - - - - - - - - - - - - - - - - - - - - -'
print(metrics.classification_report(labels[-100:], svm_pred))
print(metrics.confusion_matrix(labels[-100:], svm_pred))