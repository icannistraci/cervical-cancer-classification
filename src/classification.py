import time
import warnings

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import tree
from sklearn.neighbors import KNeighborsClassifier

from src import functions as util, models
from collections import Counter
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def warning_ignore(*args, **kwargs):
    pass


start_time = time.time()
warnings.warn = warning_ignore

print(f'\n --- START Classification --- \n')


def classify(df):
    Y = df[['Biopsy']]
    X = df.drop(['Biopsy', 'Citology', 'Schiller', 'Hinselmann'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # scale between 0 and 1
    scaled_data = util.scaling_data_multiple(X_train, X_test)

    X_train = scaled_data[1][1]
    X_test = scaled_data[1][2]

    # oversampling data
    print(f'Before SMOTEENN, the shape of X_train is {X_train.shape}')
    print(f'Before SMOTEENN, the shape of Y_train is {Y_train.shape}')
    print('Before SMOTEENN, counter for Biopsy is ', Counter(Y_train['Biopsy']))
    print()

    smote_enn = SMOTEENN(random_state=0)
    X_train, Y_train = smote_enn.fit_resample(X_train, Y_train)

    print(f'After SMOTEENN, the shape of X_train is {X_train.shape}')
    print(f'After SMOTEENN, the shape of Y_train is {Y_train.shape}')
    print('After SMOTEENN, counter for Biopsy is ', Counter(Y_train['Biopsy']))




    ######## Logistic Regression ########

    print('\n------------ LOGISTIC REGRESSION ------------\n')

    acc_best, f1_best, sens_best, c_best, penalty_best = models.logistic_regression_classifier(X_train, Y_train)
    print(f'Accuracy train: {round(acc_best, 2)}')
    print(f'F1 train: {round(f1_best, 2)}')
    print(f'Sensitivity train: {round(sens_best, 2)}')

    print(f'LR best parameters: solver=liblinear, '
          f'penalty={penalty_best}, '
          f'C={c_best}')

    log_reg = LogisticRegression(penalty=penalty_best, solver='liblinear', C=c_best)
    log_reg.fit(X_train, Y_train.values.ravel())

    Y_pred_lr = log_reg.predict(X_test)
    Y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

    lr_results = []
    lr_results.append(accuracy_score(Y_test, Y_pred_lr))
    lr_results.append(precision_score(Y_test, Y_pred_lr, average='macro'))
    lr_results.append(recall_score(Y_test, Y_pred_lr, average='macro'))
    lr_results.append(f1_score(Y_test, Y_pred_lr, average='macro'))
    lr_results.append(classification_report(Y_test, Y_pred_lr))
    lr_results.append(confusion_matrix(Y_test, Y_pred_lr))

    accuracy_lr = lr_results[0]
    precision_lr = lr_results[1]
    recall_lr = lr_results[2]
    fscore_lr = lr_results[3]
    report_lr = lr_results[4]
    cm_lr = lr_results[5]

    tn, fp, fn, tp = cm_lr.ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(f'Accuracy: {round(accuracy_lr, 2)}')
    print(f'Sensitivity: {round(recall_lr, 2)}')
    tn_rate_lr = round(tn / (tn + fp), 2)
    print(f'Specificity: {tn_rate_lr}')
    print(f'Precision: {round(precision_lr, 2)}')
    print(f'F-measure: {round(fscore_lr, 2)}')
    print(f'Report:\n{report_lr}\n')

    auc_lr = roc_auc_score(Y_test, Y_prob_lr)
    print('AUC: %.2f' % auc_lr)
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(Y_test, Y_prob_lr)
    plot_roc_curve(fpr_lr, tpr_lr, 'lr')

    ######## Decision Tree ########

    print('\n------------ DECISION TREE ------------\n')

    acc_best, f1_best, sens_best, criterion_best, splitter_best = models.decision_tree_classifier(X_train, Y_train)
    print(f'Accuracy train: {round(acc_best, 2)}')
    print(f'F1 train: {round(f1_best, 2)}')
    print(f'Sensivity train: {round(sens_best, 2)}')

    print(f'Dtree best parameters: criterion={criterion_best}, '
          f'splitter={splitter_best}')

    dtree = tree.DecisionTreeClassifier(criterion=criterion_best, splitter=splitter_best)
    dtree.fit(X_train, Y_train.values.ravel())

    Y_pred_dtree = dtree.predict(X_test)
    Y_prob_dtree = dtree.predict_proba(X_test)[:, 1]

    dtree_results = []
    dtree_results.append(accuracy_score(Y_test, Y_pred_dtree))
    dtree_results.append(precision_score(Y_test, Y_pred_dtree, average='macro'))
    dtree_results.append(recall_score(Y_test, Y_pred_dtree, average='macro'))
    dtree_results.append(f1_score(Y_test, Y_pred_dtree, average='macro'))
    dtree_results.append(classification_report(Y_test, Y_pred_dtree))
    dtree_results.append(confusion_matrix(Y_test, Y_pred_dtree))

    accuracy_dt = dtree_results[0]
    precision_dt = dtree_results[1]
    recall_dt = dtree_results[2]
    fscore_dt = dtree_results[3]
    report_dt = dtree_results[4]
    cm_dt = dtree_results[5]

    tn, fp, fn, tp = cm_dt.ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(f'Accuracy: {round(accuracy_dt, 2)}')
    print(f'Sensitivity: {round(recall_dt, 2)}')
    tn_rate_dt = round(tn / (tn + fp), 2)
    print(f'Specificity: {tn_rate_dt}')
    print(f'Precision: {round(precision_dt, 2)}')
    print(f'F-measure: {round(fscore_dt, 2)}')
    print(f'Report:\n{report_dt}\n')

    auc_dt = roc_auc_score(Y_test, Y_prob_dtree)
    print('AUC: %.2f' % auc_dt)
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(Y_test, Y_prob_dtree)
    plot_roc_curve(fpr_dt, tpr_dt, 'dt')


    ######## Support Vector Machine ########

    print('\n------------ SVM ------------\n')

    acc_best, f1_best, sens_best, c_best, gamma_best, kernel_best = models.support_vector_machine_classifier(X_train, Y_train)
    print(f'Accuracy train: {round(acc_best, 2)}')
    print(f'F1 train: {round(f1_best, 2)}')
    print(f'Sensitivity train: {round(sens_best, 2)}')

    print(f'SVM best parameters: C={c_best}, '
          f'kernel={kernel_best}, '
          f'gamma={gamma_best}')

    svm = SVC(C=c_best, gamma=gamma_best, kernel=kernel_best, probability=True)
    svm.fit(X_train, Y_train.values.ravel())

    Y_pred_svm = svm.predict(X_test)
    Y_prob_svm = svm.predict_proba(X_test)[:, 1]

    svm_results = []
    svm_results.append(accuracy_score(Y_test, Y_pred_svm))
    svm_results.append(precision_score(Y_test, Y_pred_svm, average='macro'))
    svm_results.append(recall_score(Y_test, Y_pred_svm, average='macro'))
    svm_results.append(f1_score(Y_test, Y_pred_svm, average='macro'))
    svm_results.append(classification_report(Y_test, Y_pred_svm))
    svm_results.append(confusion_matrix(Y_test, Y_pred_svm))

    accuracy_svm = svm_results[0]
    precision_svm = svm_results[1]
    recall_svm = svm_results[2]
    fscore_svm = svm_results[3]
    report_svm = svm_results[4]
    cm_svm = svm_results[5]

    tn, fp, fn, tp = cm_svm.ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(f'Accuracy: {round(accuracy_svm, 2)}')
    print(f'Sensitivity: {round(recall_svm, 2)}')
    tn_rate = round(tn / (tn + fp), 2)
    print(f'Specificity: {tn_rate}')
    print(f'Precision: {round(precision_svm, 2)}')
    print(f'F-measure: {round(fscore_svm, 2)}')
    print(f'Report:\n{report_svm}\n')

    auc_svm = roc_auc_score(Y_test, Y_prob_svm)
    print('AUC: %.2f' % auc_svm)
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(Y_test, Y_prob_svm)
    plot_roc_curve(fpr_svm, tpr_svm, 'svm')


    ######## K-Nearest Neighbors ########

    print('\n------------ KNN ------------\n')

    acc_best, f1_best, sens_best, n_best, weight_best, algo_best = models.k_nearest_neighbors_classifier(X_train, Y_train)
    print(f'Accuracy train: {round(acc_best, 2)}')
    print(f'F1 train: {round(f1_best, 2)}')
    print(f'Sensitivity train: {round(sens_best, 2)}')

    print(f'KNN best parameters: weights={weight_best}, '
          f'n_neighbors={n_best}, '
          f'algorithm={algo_best}')

    knn = KNeighborsClassifier(n_neighbors=n_best, weights=weight_best, algorithm=algo_best)
    knn.fit(X_train, Y_train.values.ravel())

    Y_pred_knn = knn.predict(X_test)
    Y_prob_knn = knn.predict_proba(X_test)[:, 1]

    knn_results = []
    knn_results.append(accuracy_score(Y_test, Y_pred_knn))
    knn_results.append(precision_score(Y_test, Y_pred_knn, average='macro'))
    knn_results.append(recall_score(Y_test, Y_pred_knn, average='macro'))
    knn_results.append(f1_score(Y_test, Y_pred_knn, average='macro'))
    knn_results.append(classification_report(Y_test, Y_pred_knn))
    knn_results.append(confusion_matrix(Y_test, Y_pred_knn))

    accuracy_knn = knn_results[0]
    precision_knn = knn_results[1]
    recall_knn = knn_results[2]
    fscore_knn = knn_results[3]
    report_knn = knn_results[4]
    cm_knn = knn_results[5]

    tn, fp, fn, tp = cm_knn.ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(f'Accuracy: {round(accuracy_knn, 2)}')
    print(f'Sensitivity: {round(recall_knn, 2)}')
    tn_rate_knn = round(tn / (tn + fp), 2)
    print(f'Specificity: {tn_rate_knn}')
    print(f'Precision: {round(precision_knn, 2)}')
    print(f'F-measure: {round(fscore_knn, 2)}')
    print(f'Report:\n{report_knn}\n')

    auc_knn = roc_auc_score(Y_test, Y_prob_knn)
    print('AUC: %.2f' % auc_knn)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(Y_test, Y_prob_knn)
    plot_roc_curve(fpr_knn, tpr_knn, 'knn')

    ######## MLP ########

    print('\n------------ MLP ------------\n')

    score_best, sens_best, f1_best, acc_best, neurons_best, actfun_best, solver_best = models.multilayer_perceptron_classifier(X_train, Y_train)
    print(f'Accuracy train: {round(acc_best, 2)}')
    print(f'F1 train: {round(f1_best, 2)}')
    print(f'Sensitivity train: {round(sens_best, 2)}')

    print(f'KNN best parameters: hidden_layer_sizes={neurons_best}, '
          f'activation={actfun_best}, '
          f'solver={solver_best}')

    mlp = MLPClassifier(hidden_layer_sizes=neurons_best, activation=actfun_best, solver=solver_best,
                        tol=10e-4, max_iter=700)
    mlp.fit(X_train, Y_train.values.ravel())

    Y_pred_mlp = mlp.predict(X_test)
    Y_prob_mlp = mlp.predict_proba(X_test)[:, 1]

    mlp_results = []
    mlp_results.append(accuracy_score(Y_test, Y_pred_mlp))
    mlp_results.append(precision_score(Y_test, Y_pred_mlp, average='macro'))
    mlp_results.append(recall_score(Y_test, Y_pred_mlp, average='macro'))
    mlp_results.append(f1_score(Y_test, Y_pred_mlp, average='macro'))
    mlp_results.append(classification_report(Y_test, Y_pred_mlp))
    mlp_results.append(confusion_matrix(Y_test, Y_pred_mlp))

    accuracy_mlp = mlp_results[0]
    precision_mlp = mlp_results[1]
    recall_mlp = mlp_results[2]
    fscore_mlp = mlp_results[3]
    report_mlp = mlp_results[4]
    cm_mlp = mlp_results[5]

    tn, fp, fn, tp = cm_mlp.ravel()
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    print(f'Accuracy: {round(accuracy_mlp, 2)}')
    print(f'Sensitivity: {round(recall_mlp, 2)}')
    tn_rate_mlp = round(tn / (tn + fp), 2)
    print(f'Specificity: {tn_rate_mlp}')
    print(f'Precision: {round(precision_mlp, 2)}')
    print(f'F-measure: {round(fscore_mlp, 2)}')
    print(f'Report:\n{report_mlp}\n')

    auc_mlp = roc_auc_score(Y_test, Y_prob_mlp)
    print('AUC: %.2f' % auc_mlp)
    fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(Y_test, Y_prob_mlp)
    plot_roc_curve(fpr_mlp, tpr_mlp, 'mlp')


    # plot all ROC together
    plt.figure(figsize=(16, 8))
    plt.title('ROC Curve \n 5 Classifiers', fontsize=18)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression Classifier')
    plt.plot(fpr_knn, tpr_knn, label=f'KNears Neighbors Classifier')
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree Classifier')
    plt.plot(fpr_dt, tpr_dt, label=f'Support Vector Machine Classifier')
    plt.plot(fpr_mlp, tpr_mlp, label=f'Multi Layer Perceptron Classifier')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1.2, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05))
    plt.legend()
    plt.show()


print(f'\n --- END Classification in {(time.time() - start_time)} seconds ---')


def plot_roc_curve(fpr, tpr, classifier):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if classifier == 'lr':
        plt.title('LR ROC Curve')
    if classifier == 'knn':
        plt.title('KNN ROC Curve')
    if classifier == 'svm':
        plt.title('SVM ROC Curve')
    if classifier == 'dt':
        plt.title('Dtree ROC Curve')
    if classifier == 'mlp':
        plt.title('MLP Curve')
    plt.legend()
    plt.show()
