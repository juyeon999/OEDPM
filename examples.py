# -*- coding: utf-8 -*-
# Toy example for OEDPM
# @Time    : 2024/1/1
# @Author  : Juyeon Park (wndus1712@gmail.com)

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
