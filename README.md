This repository is the source code of the paper "**Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures**" published in Pattern Recognition (July 2024). (see the full paper at https://arxiv.org/abs/2401.00773 or https://doi.org/10.1016/j.patcog.2024.110846) 

# How to use?
OEDPM offers user-friendly APIs similar to those in sklearn. Initially, we create an instance of the model class using specified parameters. Once instantiated, this model can then be employed for fitting and predicting data.

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

# Examples
```python
from oedpm import OEDPM
from data_generator import benchmark_generator
from common import calculate_metrics

# Generate data
X, y = benchmark_generator('musk')

# Fit the model
model_configs = {'n_components':100, 'useIQRMethod':False, 'contamination':'auto'}
model = OEDPM(**model_configs)
model.fit(X)
y_pred = model.predict(X)  # Predicted value for training data
y_scores = model.y_scores  # Outlier scores for training data

# Calculate metrics
calculate_metrics(y, y_pred, y_scores)
```
