Implement of "Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures"

# Files


# Parameters:


# Attributes:


# Methods
- `fit(X)`: Train model 
- `fit_DPGMM(X)`
- get_cluster_assignments(model, X)
- get_inlier_clusters(model, num_assignments, supervised)
- get_log_likelihood(X, mixture_weights, mixture_mus, mixture_covs)
- get_outlier_threshold_contamination(x)
- get_outlier_threshold_IQR(x)
- predict(X, voting_thres)
- random_proj(X)

  
# Examples
---
```
>>> from sklearn.ensemble import IsolationForest
>>> X = [[-1.1], [0.3], [0.5], [100]]
>>> clf = IsolationForest(random_state=0).fit(X)
>>> clf.predict([[0.1], [0], [90]])
array([ 1,  1, -1])
```

또는 
https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py

1. Requirements 파일 만들기
2. py 파일 말기
3. 예시 코드 작성
