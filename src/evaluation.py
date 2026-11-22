"""Simple top-K offline metrics for implicit relevance (rating >= threshold)."""
import pandas as pd


def precision_recall_at_k(recommended_ids, relevant_ids, k: int = 10):
    """Return precision@K and recall@K for one user; relevance is supplied by caller."""
    if k < 1:
        raise ValueError("k must be at least 1")
    recs = list(dict.fromkeys(recommended_ids))[:k]
    relevant = set(relevant_ids)
    hits = sum(movie_id in relevant for movie_id in recs)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return {f"precision@{k}": precision, f"recall@{k}": recall}


def leave_one_out_split(ratings: pd.DataFrame, random_state: int = 42):
    """Hold out each user's last timestamped rating; returns train and test frames."""
    ordered = ratings.sort_values(["userId", "timestamp"], kind="stable")
    test_idx = ordered.groupby("userId", sort=False).tail(1).index
    test = ratings.loc[test_idx].copy()
    train = ratings.drop(index=test_idx).copy()
    return train, test


def rmse(actual, predicted):
    """Root mean squared error for rating predictions."""
    import numpy as np
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be non-empty and have equal lengths")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
