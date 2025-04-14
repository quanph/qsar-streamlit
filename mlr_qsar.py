import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def mlr_qsar_metrics(X, y, intercept=True):
    n, p = X.shape
    if intercept:
        X_ = np.column_stack((np.ones(n), X))
        p += 1
    else:
        X_ = X.copy()
    # Fit model
    beta = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
    y_pred = X_ @ beta

    # Residuals & R2
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_reg = np.sum((y_pred - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = ss_reg / ss_total
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = ss_res / (n - p - 1)
    rmse = np.sqrt(mse)
    msr = ss_reg / (p - 1)
    F = msr / mse

    # Leave-One-Out Cross-Validation
    y_pred_cv = []
    for i in range(n):
        X_loo = np.delete(X_, i, axis=0)
        y_loo = np.delete(y, i)
        beta_i = np.linalg.pinv(X_loo.T @ X_loo) @ X_loo.T @ y_loo
        y_i_pred = X_[i, :] @ beta_i
        y_pred_cv.append(y_i_pred)
    y_pred_cv = np.array(y_pred_cv)
    rmsecv = np.sqrt(np.mean((y_pred_cv - y) ** 2))

    # Q2 and Fcv
    q2_num = np.sum((2 * y - y_mean - y_pred_cv) * (y_pred_cv - y_mean))
    q2_den = np.sum((y - y_mean) ** 2)
    q2 = q2_num / q2_den if q2_den != 0 else 0
    Fcv = (q2 * (n - p)) / (p * (1 - q2)) if (1 - q2) != 0 else 0

    return {
        "beta": beta,
        "pred_train": y_pred,
        "pred_loocv": y_pred_cv,
        "R2": r2,
        "adjusted_R2": adj_r2,
        "Q2": q2,
        "F": F,
        "Fcv": Fcv,
        "RMSE": rmse,
        "RMSECV": rmsecv
    }