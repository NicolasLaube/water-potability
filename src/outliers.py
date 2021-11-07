from nptyping import NDArray
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor


class OutliersDeletion(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs) -> None:
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """
        super().__init__()
        self.kwargs = kwargs
    
    def transform(self, X: NDArray, y = None):
        """Called when we use fit or transform on the pipeline"""
        lcf = LocalOutlierFactor(**self.kwargs) # Unsupervised Outlier Detection using the Local Outlier Factor (LOF).
        lcf.fit(X)
        nof = lcf.negative_outlier_factor_
        return X[nof > np.quantile(nof, 0.95), :], y[nof > np.quantile(nof, 0.95)]

    def fit(self, *args, **kwargs):
        return self
