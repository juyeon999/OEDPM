from oedpm import OEDPM
from data_generator import benchmark_generator
from common import calculate_metrics
import numpy as np

# Generate data
X_train, y = benchmark_generator('musk')

# Fit the model
np.random.seed(3920)
model_configs = {'n_components':100, 'useIQRMethod':False, 'contamination':'auto'}
model = OEDPM(**model_configs)
model.fit(X_train)
y_pred = model.predict(X_train)  # Predicted value for training data
y_scores = model.y_scores  # Outlier scores for training data

# Calculate metrics
calculate_metrics(y, y_pred, y_scores)