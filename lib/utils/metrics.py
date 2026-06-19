from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
import numpy as np

def one_dim_ssim(x, y, num_breaks=256):

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    x = x / x.max() if x.max() > 0 else np.zeros_like(x)
    y = y / y.max() if y.max() > 0 else np.zeros_like(y)

    x_dig = np.digitize(x, np.linspace(0, 1, num_breaks), right=True) - 1
    y_dig = np.digitize(y, np.linspace(0, 1, num_breaks), right=True) - 1

    C1 = (0.01 * (num_breaks - 1)) ** 2
    C2 = (0.03 * (num_breaks - 1)) ** 2
    mux = x_dig.mean()
    muy = y_dig.mean()
    sigx = x_dig.var()
    sigy = y_dig.var()
    sigxy = np.cov(x_dig, y_dig, ddof=0)[0, 1]
    ssim = ((2 * mux * muy + C1) * (2 * sigxy + C2)) / ((mux ** 2 + muy ** 2 + C1) * (sigx + sigy + C2))
    return float(ssim)

def compute_gene_metrics(true, pred, binary_threshold=0, log_space=True, metrics=None):
    if metrics is None:
        metrics = ['R2', 'PCC', 'SCC', 'RMSE', 'AUC']

    results = {}

    if log_space:
        true_raw = np.expm1(true)
        pred_raw = np.expm1(pred)
    else:
        true_raw = true
        pred_raw = pred

    if 'R2' in metrics:
        results['R2'] = float(r2_score(true, pred))
    if 'PCC' in metrics:
        results['PCC'] = float(pearsonr(true, pred)[0])
    if 'SCC' in metrics:
        results['SCC'] = float(spearmanr(true, pred)[0])

    if 'MI' in metrics:
        try:
            results['MI'] = float(mutual_info_regression(pred.reshape(-1, 1), true)[0])
        except:
            results['MI'] = np.nan

    if 'JS_Div' in metrics:
        def js_div(p, q):
            p = np.clip(p, 1e-10, None)
            q = np.clip(q, 1e-10, None)
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        p_norm = true / np.sum(true)
        q_norm = pred / np.sum(pred)
        results['JS_Div'] = float(js_div(p_norm, q_norm))

    if 'RMSE' in metrics:
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        results['RMSE'] = float(rmse)

    if 'AUC' in metrics:
        try:
            true_bin = (true_raw > binary_threshold).astype(int)
            results['AUC'] = float(roc_auc_score(true_bin, pred_raw))
        except:
            results['AUC'] = np.nan

    if 'SSIM' in metrics:
        try:
            results['SSIM'] = one_dim_ssim(true_raw, pred_raw)
        except:
            results['SSIM'] = float('nan')

    if 'MAE' in metrics:
        results['MAE'] = float(np.mean(np.abs(pred - true)))

    for key in results:
        if isinstance(results[key], np.floating) and np.isnan(results[key]):
            results[key] = float('nan')

    return results
