from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


def logistic_regression_classifier(X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    k = 0
    f1_best = 0
    acc_best = 0
    c_best = 0
    sens_best = 0
    penalty_best = 0
    poss_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    poss_penalty = ['l2', 'l1']
    # poss_solver = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
    for c in poss_C:
        for penalty in poss_penalty:
            y_pred = np.zeros(len(y))
            y_prob = np.zeros(len(y))
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

                clf = LogisticRegression(penalty=penalty, solver='liblinear', C=c)
                clf.fit(X_train, y_train.values.ravel())

                y_pred[test_index] = clf.predict(X_test)
                y_prob[test_index] = clf.predict_proba(X_test)[:, 1]

            f1 = f1_score(y, y_pred)
            acc = accuracy_score(y, y_pred)
            sens = recall_score(y, y_pred, average='macro')

            print(f'LR: k={k}, f1={f1}, sensitivity={sens}, penalty={penalty}, C={c}')

            if sens > sens_best:
                f1_best = f1
                acc_best = acc
                sens_best = sens
                c_best = c
                penalty_best = penalty

            k += 1

    print(f'Best parameters are: penalty={penalty_best} and C={c_best}')

    return acc_best, f1_best, sens_best, c_best, penalty_best


def decision_tree_classifier(X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    k = 0
    acc_best = 0
    f1_best = 0
    sens_best = 0
    criterion_best = 0
    splitter_best = 0
    poss_criterion = ['gini', 'entropy']
    poss_splitter = ['best', 'random']
    for criterion in poss_criterion:
        for splitter in poss_splitter:
            y_pred = np.zeros(len(y))
            y_prob = np.zeros(len(y))
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

                clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter)
                clf.fit(X_train, y_train.values.ravel())

                y_pred[test_index] = clf.predict(X_test)
                y_prob[test_index] = clf.predict_proba(X_test)[:, 1]

            f1 = f1_score(y, y_pred)
            acc = accuracy_score(y, y_pred)
            sens = recall_score(y, y_pred, average='macro')

            print(f'Dtree: k={k}, f1={f1}, sensitivity={sens}, criterion={criterion}, splitter={splitter}')

            if sens > sens_best:
                f1_best = f1
                acc_best = acc
                sens_best = sens
                criterion_best = criterion
                splitter_best = splitter

            k += 1

    print(f'Best parameters are: criterion={criterion_best} and splitter={splitter}')

    return acc_best, f1_best, sens_best, criterion_best, splitter_best


def support_vector_machine_classifier(X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    k = 0
    acc_best = 0
    f1_best = 0
    sens_best = 0
    c_best = 0
    kernel_best = 0
    gamma_best = 0
    poss_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    poss_c = [0.1, 1, 10, 100, 1000]
    poss_gamma = ['scale', 'auto']
    for gamma in poss_gamma:
        for C in poss_c:
            for kernel in poss_kernel:
                y_pred = np.zeros(len(y))
                y_prob = np.zeros(len(y))
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

                    clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
                    clf.fit(X_train, y_train.values.ravel())

                    y_pred[test_index] = clf.predict(X_test)
                    y_prob[test_index] = clf.predict_proba(X_test)[:, 1]

                f1 = f1_score(y, y_pred)
                acc = accuracy_score(y, y_pred)
                sens = recall_score(y, y_pred, average='macro')

                print(f'SVC: k={k}, f1={f1}, sensitivity={sens}, kernel={kernel}, C={C}, gamma={gamma}')

                if sens > sens_best:
                    f1_best = f1
                    acc_best = acc
                    sens_best = sens
                    c_best = C
                    kernel_best = kernel
                    gamma_best = gamma

                k += 1

    print(f'Best parameters are: gamma={gamma_best}, kernel={kernel_best}, C={c_best}')

    return acc_best, f1_best, sens_best, c_best, gamma_best, kernel_best


def k_nearest_neighbors_classifier(X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    k = 0
    acc_best = 0
    f1_best = 0
    sens_best = 0
    n_best = 0
    weight_best = 0
    algo_best = 0
    poss_weights = ['uniform', 'distance']
    poss_n = [5, 10, 15, 20, 25]
    poss_algo = ['auto', 'brute', 'ball_tree', 'kd_tree']
    for n in poss_n:
        for algo in poss_algo:
            for weight in poss_weights:
                y_pred = np.zeros(len(y))
                y_prob = np.zeros(len(y))
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

                    clf = KNeighborsClassifier(n_neighbors=n, weights=weight, algorithm=algo)
                    clf.fit(X_train, y_train.values.ravel())

                    y_pred[test_index] = clf.predict(X_test)
                    y_prob[test_index] = clf.predict_proba(X_test)[:, 1]

                f1 = f1_score(y, y_pred)
                acc = accuracy_score(y, y_pred)
                sens = recall_score(y, y_pred, average='macro')

                print(f'KNN: k={k}, accuracy={acc}, sensitivity={sens}, n_neighbors={n}, weights={weight}, algorithm={algo}')

                if sens > sens_best:
                    f1_best = f1
                    acc_best = acc
                    sens_best = sens
                    n_best = n
                    weight_best = weight
                    algo_best = algo

                k += 1

    print(f'Best parameters are: n_neighbors={n_best}, weights={weight_best}, algorithm={algo_best}')

    return acc_best, f1_best, sens_best, n_best, weight_best, algo_best


def multilayer_perceptron_classifier(X, y):
    kf = KFold(n_splits=5, shuffle=False, random_state=42)

    k = 0
    f1_best = 0
    acc_best = 0
    solver_best = 0
    score_best = 0
    sens_best = 0
    neurons_best = 0
    actfun_best = 0
    poss_neurons = np.arange(20, 30, 2, dtype=int)
    poss_actfun = ['logistic', 'relu', 'tanh']
    poss_solver = ['lbfgs', 'sgd', 'adam']
    for neurons in poss_neurons:
        for actfun in poss_actfun:
            for solver in poss_solver:
                y_pred = np.zeros(len(y))
                y_prob = np.zeros(len(y))
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

                    clf = MLPClassifier(hidden_layer_sizes=neurons, activation=actfun, solver=solver,
                                         random_state=42, tol=10e-4, max_iter=700)
                    clf.fit(X_train, y_train)

                    y_pred[test_index] = clf.predict(X_test)
                    y_prob[test_index] = clf.predict_proba(X_test)[:, 1]

                f1 = f1_score(y, y_pred)
                acc = accuracy_score(y, y_pred)
                sens = recall_score(y, y_pred, average='macro')

                print(f'MLP: k={k}, neurons_best={neurons}, activation={actfun}, solver={solver}')

                if sens > sens_best:
                    acc_best = acc
                    sens_best = sens
                    neurons_best = neurons
                    actfun_best = actfun
                    solver_best = solver
                    f1_best = f1

                k += 1

    return score_best, sens_best, f1_best, acc_best, neurons_best, actfun_best, solver_best
