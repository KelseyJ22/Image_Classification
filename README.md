# Image_Classification
Fashion MNIST dataset

BASELINE
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
Logistic Regression Results:
- - - - - - - - - - - - - - - - - - - - - - - - - - - -
             precision    recall  f1-score   support

          0       0.62      0.71      0.67         7
          1       0.86      1.00      0.92         6
          2       0.75      0.67      0.71         9
          3       0.70      1.00      0.82         7
          4       0.62      0.71      0.67         7
          5       1.00      0.78      0.88        18
          6       0.88      0.58      0.70        12
          7       0.85      1.00      0.92        11
          8       1.00      1.00      1.00        15
          9       0.78      0.88      0.82         8

avg / total       0.85      0.83      0.83       100

[[ 5  0  0  2  0  0  0  0  0  0]
 [ 0  6  0  0  0  0  0  0  0  0]
 [ 0  0  6  0  2  0  1  0  0  0]
 [ 0  0  0  7  0  0  0  0  0  0]
 [ 0  1  0  1  5  0  0  0  0  0]
 [ 0  0  1  0  0 14  0  1  0  2]
 [ 3  0  1  0  1  0  7  0  0  0]
 [ 0  0  0  0  0  0  0 11  0  0]
 [ 0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  0  0  1  0  7]]
GaussianNB(priors=None)
Naive Bayes Results:
- - - - - - - - - - - - - - - - - - - - - - - - - - - -
             precision    recall  f1-score   support

          0       0.60      0.43      0.50         7
          1       0.55      1.00      0.71         6
          2       0.50      0.33      0.40         9
          3       0.50      0.71      0.59         7
          4       0.31      0.71      0.43         7
          5       1.00      0.28      0.43        18
          6       1.00      0.08      0.15        12
          7       0.48      1.00      0.65        11
          8       0.86      0.80      0.83        15
          9       0.67      0.75      0.71         8

avg / total       0.71      0.57      0.53       100

[[ 3  0  1  2  1  0  0  0  0  0]
 [ 0  6  0  0  0  0  0  0  0  0]
 [ 0  0  3  1  3  0  0  0  2  0]
 [ 0  2  0  5  0  0  0  0  0  0]
 [ 0  1  0  1  5  0  0  0  0  0]
 [ 0  0  0  0  0  5  0 10  0  3]
 [ 2  2  1  1  5  0  1  0  0  0]
 [ 0  0  0  0  0  0  0 11  0  0]
 [ 0  0  1  0  2  0  0  0 12  0]
 [ 0  0  0  0  0  0  0  2  0  6]]
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
K-Nearest Neighbors Results:
- - - - - - - - - - - - - - - - - - - - - - - - - - - -
             precision    recall  f1-score   support

          0       0.56      0.71      0.63         7
          1       0.75      1.00      0.86         6
          2       0.80      0.89      0.84         9
          3       0.86      0.86      0.86         7
          4       0.75      0.43      0.55         7
          5       1.00      0.83      0.91        18
          6       0.82      0.75      0.78        12
          7       0.92      1.00      0.96        11
          8       1.00      0.93      0.97        15
          9       0.70      0.88      0.78         8

avg / total       0.85      0.84      0.84       100

[[ 5  0  1  1  0  0  0  0  0  0]
 [ 0  6  0  0  0  0  0  0  0  0]
 [ 1  0  8  0  0  0  0  0  0  0]
 [ 1  0  0  6  0  0  0  0  0  0]
 [ 0  2  1  0  3  0  1  0  0  0]
 [ 0  0  0  0  0 15  0  0  0  3]
 [ 2  0  0  0  1  0  9  0  0  0]
 [ 0  0  0  0  0  0  0 11  0  0]
 [ 0  0  0  0  0  0  1  0 14  0]
 [ 0  0  0  0  0  0  0  1  0  7]]
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
Decision Tree Results:
- - - - - - - - - - - - - - - - - - - - - - - - - - - -
             precision    recall  f1-score   support

          0       0.33      0.29      0.31         7
          1       0.86      1.00      0.92         6
          2       0.67      0.67      0.67         9
          3       0.71      0.71      0.71         7
          4       0.50      0.57      0.53         7
          5       1.00      0.94      0.97        18
          6       0.46      0.50      0.48        12
          7       0.90      0.82      0.86        11
          8       1.00      0.87      0.93        15
          9       0.70      0.88      0.78         8

avg / total       0.76      0.75      0.75       100

[[ 2  0  1  1  0  0  3  0  0  0]
 [ 0  6  0  0  0  0  0  0  0  0]
 [ 0  0  6  0  2  0  1  0  0  0]
 [ 0  0  0  5  0  0  2  0  0  0]
 [ 0  1  0  1  4  0  1  0  0  0]
 [ 0  0  0  0  0 17  0  0  0  1]
 [ 2  0  2  0  2  0  6  0  0  0]
 [ 0  0  0  0  0  0  0  9  0  2]
 [ 2  0  0  0  0  0  0  0 13  0]
 [ 0  0  0  0  0  0  0  1  0  7]]
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Support Vector Machine Results:
- - - - - - - - - - - - - - - - - - - - - - - - - - - -
             precision    recall  f1-score   support

          0       0.62      0.71      0.67         7
          1       0.86      1.00      0.92         6
          2       0.86      0.67      0.75         9
          3       0.70      1.00      0.82         7
          4       0.67      0.57      0.62         7
          5       1.00      0.83      0.91        18
          6       0.80      0.67      0.73        12
          7       0.90      0.82      0.86        11
          8       1.00      1.00      1.00        15
          9       0.67      1.00      0.80         8

avg / total       0.85      0.83      0.83       100

[[ 5  0  0  2  0  0  0  0  0  0]
 [ 0  6  0  0  0  0  0  0  0  0]
 [ 0  0  6  0  2  0  1  0  0  0]
 [ 0  0  0  7  0  0  0  0  0  0]
 [ 0  1  0  1  4  0  1  0  0  0]
 [ 0  0  0  0  0 15  0  1  0  2]
 [ 3  0  1  0  0  0  8  0  0  0]
 [ 0  0  0  0  0  0  0  9  0  2]
 [ 0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  0  0  0  0  8]]

 CONVOLUTIONAL NEURAL NETWORK
 ![Alt text](relative/path/to/accuracy.png?raw=true "Accuracy")