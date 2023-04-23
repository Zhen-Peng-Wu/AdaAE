import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def UnsupervisedClassifier(embeddings, labels, search=True):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc_std = 0 # std
    acc_val = 0 # acc

    _acc_val, _acc_std = svc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val # acc
        acc_std = _acc_std # std

    return acc_val, acc_std

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

    accuracies = np.array(accuracies)
    return accuracies.mean(), accuracies.std()
