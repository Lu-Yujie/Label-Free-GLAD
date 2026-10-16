import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn import metrics, preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, accuracy_score, normalized_mutual_info_score, adjusted_rand_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import torch.nn as nn

############################
# Evaluation Functions
############################

def cal_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian (linear_sum_assignment) algorithm 
    to find the best label permutation that maximizes the matching between predicted and true labels.
    
    Parameters:
        y_true (np.array): true labels of shape (n_samples,)
        y_pred (np.array): predicted labels of shape (n_samples,)
    
    Returns:
        float: clustering accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1

    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(ind_row, ind_col)) * 1.0 / y_pred.size


def eva_clustering(K, y_true):
    """
    Evaluate clustering performance using spectral clustering on the precomputed kernel matrix K.
    
    - Perform spectral clustering with number of clusters = number of unique labels.
    - Compute accuracy, NMI, and ARI for the clustering result.
    - Accuracy is computed with label alignment, while NMI and ARI are label-invariant.
    
    Parameters:
        K (np.array): precomputed kernel matrix of shape (N, N)
        y_true (np.array): true labels of shape (N,)
    
    Returns:
        dict: {'clu_acc': accuracy, 'clu_nmi': nmi, 'clu_ari': ari}
    """
    y_pred = SpectralClustering(n_clusters=len(np.unique(y_true)), 
                                random_state=0,
                                affinity='precomputed').fit_predict(K)

    acc_score = cal_acc(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)

    return {'clu_acc': acc_score, 'clu_nmi': nmi, 'clu_ari': ari}


def eva_svc(K, y_true):
    """
    Evaluate classification performance using SVM with a precomputed kernel (K).
    Perform a 10-fold cross-validation and compute mean and std of accuracies.
    
    Parameters:
        K (np.array): precomputed kernel matrix (N x N)
        y_true (np.array): true labels (N,)
    
    Returns:
        dict: {'cv_svc_mean': mean_accuracy, 'cv_svc_std': std_accuracy}
    """
    clf = SVC(kernel="precomputed", tol=1e-6, probability=True)
    acc_score = cross_val_score(clf, K, y_true, cv=10)
    return {'cv_svc_mean': np.mean(acc_score), 'cv_svc_std': np.std(acc_score)}


class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)
    

def logistic_classify(x, y, device="cpu"):
    num_classes = np.unique(y).shape[0]
    num_features = x.shape[1]

    accuracies = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegressionModel(num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            outputs = model(x_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracies.append(correct / y_test_tensor.size(0))

    return np.mean(accuracies), np.std(accuracies)


def svc_classify(x, y, search):
    """
    Evaluate embeddings using an SVM classifier (optionally with hyperparam search) in a 10-fold CV.
    
    Parameters:
        x (np.array): embeddings (N, D)
        y (np.array): labels (N,)
        search (bool): if True, use GridSearchCV to tune C parameter
    
    Returns:
        mean_accuracy, std_accuracy
    """
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring="accuracy", verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)


def evaluate_K_unsupervised(K, labels, search=True, device="cpu"):
    """
    Evaluate the given kernel matrix K (and corresponding labels) using both:
    1. Treating K as "embeddings" by using x = K and training classifiers directly on it.
    2. Treating K as "kernel" (precomputed) and performing spectral clustering and SVC with precomputed kernel.
    
    Steps:
    - Ensure labels are encoded.
    - Convert K to CPU numpy arrays for sklearn (if on GPU).
    - logistic_classify & svc_classify treat K as raw features (x).
    - eva_clustering & eva_svc treat K as kernel.
    
    Parameters:
        K (torch.Tensor or np.ndarray): kernel matrix [N,N]
        labels (torch.Tensor or np.ndarray): labels [N]
        search (bool): whether to do grid search in svc_classify
        device (str): 'cpu' or 'cuda' - used for logistic_classify training.
    
    Returns:
        dict: containing metrics of different evaluations.
    """
    # Ensure labels and K in cpu numpy arrays for sklearn
    if torch.is_tensor(K):
        K = K.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
            
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    # x, y = np.array(K), np.array(labels)

    # Evaluate K as embeddings
    logreg_accuracy, logreg_std = logistic_classify(K, labels, device=device)
    svc_accuracy, svc_std = svc_classify(K, labels, search)

    # Evaluate K as kernel (precomputed)
    clustering_metrics = eva_clustering(K, labels)
    kernel_svc_metrics = eva_svc(K, labels)

    metrics_results = {
       'LogReg_ACC': logreg_accuracy,
       'LogReg_STD': logreg_std,
       'SVC_ACC': svc_accuracy,
       'SVC_STD': svc_std,
       'K_Clustering_ACC': clustering_metrics['clu_acc'],
       'K_NMI': clustering_metrics['clu_nmi'],
       'K_ARI': clustering_metrics['clu_ari'],
       'K_SVC_ACC': kernel_svc_metrics['cv_svc_mean'],
       'K_SVC_STD': kernel_svc_metrics['cv_svc_std']
    }

    #    'AUROC': roc_auc_score(labels, K.diag().cpu().numpy()),  # Using diagonal as scores
    #    'AUPR': average_precision_score(labels, K.diag().cpu().numpy()),
    #    'F1-Score': f1_score(labels, (K.diag().cpu().numpy() >= 0.5).astype(int)),
    #    'Precision': precision_score(labels, (K.diag().cpu().numpy() >= 0.5).astype(int)),
    #    'Recall': recall_score(labels, (K.diag().cpu().numpy() >= 0.5).astype(int)),
    #    'FPR': metrics.false_positive_rate(labels, (K.diag().cpu().numpy() >= 0.5).astype(int)),
    #    'FNR': metrics.false_negative_rate(labels, (K.diag().cpu().numpy() >= 0.5).astype(int)),
       
    return metrics_results



def calculate_fpr95(y_true, y_score):
    """
    Calculate FPR at TPR 95%
    Args:
        y_true: Ground truth labels (0: normal, 1: anomaly)
        y_score: Anomaly scores (higher score = more anomalous)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    target_tpr = 0.95
    diff = np.abs(tpr - target_tpr)
    idx = np.argmin(diff)
    
    return fpr[idx]