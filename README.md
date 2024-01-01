This repository is the source code of the paper "**Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures**". (see the full paper at xxx) 
The model proposed in this paper is referred to as the **Outlier Ensemble of Dirichlet Process Mixtures (OEDPM)**.

# How to use?
OEDPM provides easy APIs like the sklearn style. We first instantiate the model class by giving the parameters then, the instantiated model can be used to fit and predict data.
```python
from oedpm import OEDPM
model_configs = {'n_components':100, 'useIQRMethod':False, 'contamination':'auto'}
model = OEDPM(**model_configs)
model.fit(X_train)
y_pred = model.predict(X_train)  # Predicted value for training data
y_scores = model.y_scores  # Outlier scores for training data
```

# Parameters
- `n_components`: Number of parallel estimators in the ensemble. Default is 100.  
- `cov_type`: Covariance restriction of the Gaussian mixture. Default is 'diag'.  
- `useIQRMethod`: Indicates whether to use the IQR method or contamination methods for outlier detection. Default is True.  
  - If True, use the `whis` parameter.
  - If False, use the `contamination` parameter.
- `whis`: Whisker length in the IQR method. Default is 1.5.
  - This is used only when `useIQRMethod` is True.
- `contamination`: The proportion of outliers in the dataset, used to set the threshold for outlier detection when not using the IQR method. Default is 'auto'.
  - This is used only when `useIQRMethod` is False.
- `supervised`: Boolean indicating whether the training dataset consists only of normal classes. If true, inlier cluster selection is not necessary. Default is False.

# Methods
- `fit(X)`: Fit estimator.
- `predict(X)`: Predict whether a particular sample is an outlier.
- `random_proj(X)`: Random projection and subsampling.
- `fit_DPGMM(X)`: Fit DPGMM using variational inference
- `get_cluster_assignments(model, X)`: Allocate data points by cluster based on fitted DPGMM.
- `get_inlier_clusters(model, num_assignments, supervised)`: Select inlier cluster based on cluster weights.
- `get_log_likelihood(X, mixture_weights, mixture_mus, mixture_covs)`: Calculate the log likelihood of all data points. Instead of replacing everything with infinity, use log-sum-exp tricks to eliminate underflow. 
- `get_outlier_threshold_contamination(x)`: Calculates the outlier threshold based on the contamination level. It sorts the data in 'x' and finds the value at the percentile corresponding to `contamination`. This value is then returned as the threshold for identifying outliers.
- `get_outlier_threshold_IQR(x)`: Calculates the outlier threshold using the IQR method. The lower bound is computed as Q1 minus `whis` times the IQR. The threshold is set at the smallest value in 'x' that is less than this lower bound.

# Returns
- `y_pred`: For each observation, it tells whether or not (1 or 0) it should be considered as an outlier according to the fitted model.
- `y_scores`: Outlier score of X of the base classifiers. The outlier score of an input sample is computed as the mean of binarized likelihood values from each ensemble component.

