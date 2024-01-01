# -*- coding: utf-8 -*-
# Implementation of Outlier Ensemble of Dirichlet Process Mixtures
# @Time    : 2024/1/1
# @Author  : Kim, Dongwook and Juyeon Park (wndus1712@gmail.com)

import numpy as np
import pandas as pd
import math 
import scipy
import warnings
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

warnings.filterwarnings("ignore")

def gramschmidt(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R

class OEDPM():
    """ Class of Outlier Ensemble of Dirichlet Process Mixture (OEDPM)
    
    Parameters
    ----------
        n_components: int, optional (default=100)
            Number of parallel estimators in the ensemble.
        cov_type: str, optional (default='diag')
            Covariance restriction of the gaussian mixture.
        useIQRMethod: bool, optional (default=True)
            Indicates whether to use the IQR method or use contamination methods.
            - If True, the 'whis' parameter is used.
            - If False, the 'contamination' parameter is used.
        whis: float, optional (default=1.5)
            Whisker length in the IQR method. Used only when useIQRMethod=True.
        contamination: float, optional (default='auto')
            The proportion of outliers in the dataset, used to set the threshold for outlier detection when not using the IQR method (i.e., useIQRMethod=False).
            The 'auto' means contamination = 0.1.
        supervised: bool, optional (default=False)
            Boolean of whether the train dataset only consists of normal class. If so, inlier cluster selection is not necessary.
     
    Notes
    -----
        The 'contamination' parameter is used only when useIQRMethod=False.
        The 'whis' parameter is used only when useIQRMethod=True.

    """
    def __init__(
            self,
            n_components=100,
            cov_type='diag',
            useIQRMethod=True,
            whis=1.5,
            contamination='auto',
            supervised=False
            ):
        
        self.n_components = n_components
        self.cov_type = cov_type
        self.useIQRMethod = useIQRMethod
        if self.useIQRMethod:
            self.whis = whis
        else:
            self.contamination = 0.1 if contamination=='auto' else contamination
        self.supervised = supervised
                
    def random_proj(self, X):
        """
        Random rotated projection (Outlier Analysis, Aggarwal, Charu C, 2017 p.202)
        """
        # Random projection dimension
        lower = 2 + math.ceil(np.sqrt(X.shape[1])/2)  # reduced dimension = 2 + sqrt(d)/2
        upper = 2 + math.floor(np.sqrt(X.shape[1]))     # reduced dimension = 2 + sqrt(d)  
        d = np.random.randint(lower, upper+1) if X.shape[1] > 2 else 2  # bigger than 2

        # Random rotatation system
        Y_ = np.random.uniform(-1, 1, size=(X.shape[1], d))
        R, _ = gramschmidt(Y_)  # Gram-Schmidt orthogonalization
        X_reduced = np.matmul(np.array(X), R)  

        # Variable Subsampling
        N = len(X_reduced)
        if N < 1000:
            rand_idx = [i for i in range(N)]
            pass
        else:
            n = np.floor(np.random.uniform(N//20, N//10)).astype(int)
            rand_idx = np.random.choice(np.arange(N), n, replace=False)      # subsample 5~10% of data
            X_reduced = X_reduced[rand_idx, :]

        return R, X_reduced

    def fit_DPGMM(self, X):
        """
        Fit DPGMM using variational inference
        """
        dpgmm = BayesianGaussianMixture(n_components=30, covariance_type=self.cov_type, init_params="k-means++", weight_concentration_prior_type="dirichlet_process")
        dpgmm.fit(X)
        return dpgmm

    def get_cluster_assignments(self, model, X):
        """
        Allocate data points by cluster based on fitted DPGMM
        """
        y_pred = model.predict(X)
        num_assignments = pd.Series(y_pred).value_counts()
        sorted_idx = np.argsort(model.weights_[num_assignments.index])[::-1]  # 배정 개수대로 내림차순 정렬
        num_assignments = num_assignments.iloc[sorted_idx] 
        return num_assignments

    def get_inlier_clusters(self, model, num_assignments, supervised=False):
        """
        Select inlier cluster based on cluster weights.
        """
        # In the case of only one fitted cluster (white noise).
        if len(num_assignments.index) == 1:
            inlier_idx = num_assignments.index.astype(int)
            return inlier_idx

        # inlier cluster selection
        if supervised:
            cluster_thres = 0.0001
        else:
            cluster_thres = (1/len(num_assignments.index)) * 1
        inlier_idx = np.where(model.weights_ > cluster_thres)[0]

        # When there are no inlier clusters
        inlier_idx = [num_assignments.index[0]] if len(inlier_idx) == 0 else inlier_idx

        return inlier_idx

    def get_log_likelihood(self, X, mixture_weights, mixture_mus, mixture_covs):
        """
        Calculate the log likelihood of all data points.
        Instead of replacing everything with infinity, use log-sum-exp tricks to eliminate underflow. 
        """
        N, _ = X.shape
        K = len(mixture_weights)  # K is the number of components in the mixture
        log_likelihoods = np.zeros((N, K))

        # Calculate the log-likelihood for each component for each data point
        for k in range(K):
            log_likelihoods[:, k] = np.log(mixture_weights[k]) + multivariate_normal.logpdf(X, mean=mixture_mus[k], cov=mixture_covs[k,:,:], allow_singular=True)
        
        # Use logsumexp to calculate the log-likelihood for the mixture
        log_likelihood = logsumexp(log_likelihoods, axis=1)
        
        # Return the total log-likelihood by summing over all data points
        return log_likelihood
    
    def get_outlier_threshold_contamination(self, x):
        """Calculates the outlier threshold based on the contamination level. 
        It sorts the data in 'x' and finds the value at the percentile corresponding to 'self.contamination'. 
        This value is then returned as the threshold for identifying outliers."""

        sorted = x.argsort()
        idx = int(np.round(len(sorted) * self.contamination))
        outlier_threshold = x[sorted[idx]]

        return outlier_threshold
    
    def get_outlier_threshold_IQR(self, x):
        """Calculates the outlier threshold using the IQR method. 
        The lower bound is computed as Q1 minus 'self.whis' times the IQR. 
        The threshold is set at the smallest value in 'x' that is less than this lower bound."""

        q1 = np.quantile(x, 0.25)
        iqr_value = scipy.stats.iqr(x)
        lower = q1 - self.whis * iqr_value

        sorted = np.sort(x)
        idx = sum(sorted < lower)
        outlier_threshold = sorted[idx]

        return outlier_threshold

    def fit(self, X):
        """
        Train model 
        """
        self.df = pd.DataFrame(X).reset_index(drop=True)
        self.rotation_system = dict()                            # random rotation systems (axis)
        self.model_params = dict()                               # DPGM parameters for each bagging instances
        self.outlier_threshold = np.empty(self.n_components)      # outlier threshold (log-likelihood) for each bagging instances
        self.train_log_likelihood = dict()
        self.train_system = dict()  # for model checking, not necessary for fit the model
        self.train_dpgmm = dict()   # for model checking, not necessary for fit the model

        # Rotated Bagging
        for i in range(self.n_components):
            ## 0. Rotated Projection
            R, X_train = self.random_proj(X)
            self.rotation_system[i] = R
            self.train_system[i] = X_train

            ## 1. Fit DPGMM
            dpgmm = self.fit_DPGMM(X_train)
            self.train_dpgmm[i] = dpgmm

            ## 2. Assign data points to clusters
            num_assignments = self.get_cluster_assignments(dpgmm, X_train)

            ## 3. inlier selection
            inlier_idx = self.get_inlier_clusters(dpgmm, num_assignments, supervised=self.supervised)
            
            ## 4. calculate log-likelihood
            mixture_weights = dpgmm.weights_[inlier_idx]
            mixture_mus = dpgmm.means_[inlier_idx]
            mixture_covs = np.array([np.eye(X_train.shape[1]) * dpgmm.covariances_[idx] for idx in inlier_idx])
            self.model_params[i] = {"weights":mixture_weights, "mus":mixture_mus, "covs":mixture_covs}
            log_likelihood = self.get_log_likelihood(X_train, mixture_weights, mixture_mus, mixture_covs)
            self.train_log_likelihood[i] = log_likelihood
            
            ## 5. outlier threshold 계산
            if self.useIQRMethod:
                self.outlier_threshold[i] = self.get_outlier_threshold_IQR(log_likelihood)
            else:
                self.outlier_threshold[i] = self.get_outlier_threshold_contamination(log_likelihood)

    def predict(self, X, voting_thres=0.5):
        """
        Predict outlier label (outlier: 1 // normal: 0)
        """
        self.voting_thres = voting_thres
        self.test_log_likelihood = dict()
        y_pred = np.zeros(shape=(1, len(X)))
        for i in range(self.n_components):
            # Rotated projection
            X_ = np.matmul(np.array(X), self.rotation_system[i])

            # Calculate log-likelihood
            score = self.get_log_likelihood(X_, self.model_params[i]["weights"], self.model_params[i]["mus"], self.model_params[i]["covs"])
            self.test_log_likelihood[i] = score
            # Define outlier
            y_pred += (score <= self.outlier_threshold[i]).astype(int)

        # Hard voting
        y_pred = y_pred.reshape(-1)
        self.y_scores = y_pred / self.n_components
        y_pred = (self.y_scores > self.voting_thres).astype(int)
        
        return y_pred
    
