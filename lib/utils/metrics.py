from scipy.stats import pearsonr
from sklearn.metrics import r2_score


SUPPORTED_GENE_METRICS = ("R2", "PCC")


def validate_gene_metrics(metrics=None):
    requested = list(SUPPORTED_GENE_METRICS if metrics is None else metrics)
    unsupported = sorted(set(requested) - set(SUPPORTED_GENE_METRICS))
    if unsupported:
        raise ValueError(
            f"Unsupported gene-prediction metrics: {unsupported}. "
            f"Supported metrics: {list(SUPPORTED_GENE_METRICS)}"
        )
    return requested


def compute_gene_metrics(true, pred, metrics=None):
    requested = validate_gene_metrics(metrics)
    results = {}

    for metric in requested:
        if metric == "R2":
            results[metric] = float(r2_score(true, pred))
        elif metric == "PCC":
            results[metric] = float(pearsonr(true, pred)[0])

    return results
