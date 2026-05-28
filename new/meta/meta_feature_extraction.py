import numpy as np
from scipy.stats import entropy

def extract_meta_features(X, y):
    meta = {}

    meta["num_instances"] = X.shape[0]
    meta["num_features"] = X.shape[1]
    meta["num_classes"] = len(np.unique(y))

    meta["mean_variance"] = np.mean(np.var(X, axis=0))

    # Correlation
    if X.shape[1] > 1:
        try:
          corr = np.corrcoef(X, rowvar=False)
          meta["mean_correlation"] = np.nanmean(np.abs(corr))
        except:
            meta["mean_correlation"] = 0
    else:
        meta["mean_correlation"] = 0

    # ✅ NEW: Class entropy (VERY IMPORTANT)
    class_counts = np.bincount(y)
    probs = class_counts / len(y)
    meta["class_entropy"] = entropy(probs)

    return meta