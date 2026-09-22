import numpy as np


def binary_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    loss = -np.mean(labels * np.log(scores + 1e-12) + (1-labels) * np.log(1-scores + 1e-12))
    positives = int(labels.sum()); negatives = len(labels) - positives
    if not positives or not negatives:
        auc = float("nan")
    else:
        order = np.argsort(scores, kind="stable")
        values = scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
        ends = np.r_[starts[1:], len(values)]
        ranks = np.empty(len(labels), dtype=float)
        ranks[order] = np.repeat((starts + 1 + ends) / 2, ends-starts)
        auc = float((ranks[labels == 1].sum() - positives*(positives+1)/2)/(positives*negatives))
    return {"auc": auc, "logloss": float(loss), "accuracy": float(np.mean((scores >= .5) == labels))}

