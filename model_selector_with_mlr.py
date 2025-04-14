from mlr_qsar import mlr_qsar_metrics  # Imported for MLR integration
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def train_and_select_model(X_train, y_train, cv=5, scoring='r2'):
    """
    Huấn luyện nhiều mô hình hồi quy và chọn mô hình có điểm Cross-Validated R² cao nhất

    Args:
        X_train (array): ma trận đặc trưng đã chuẩn hóa
        y_train (array): vector mục tiêu
        cv (int): số fold cross-validation
        scoring (str): metric đánh giá, mặc định là 'r2'

    Returns:
        dict: gồm model tốt nhất và bảng kết quả của tất cả model
    """
    models = {
        "RandomForest": RandomForestRegressor(),
        "SVR": SVR(),
        "Ridge": Ridge(),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = {}
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            results[name] = {
                "CV Mean R2": scores.mean(),
                "CV Std R2": scores.std(),
                "Model": model
            }
        except Exception as e:
            results[name] = {
                "CV Mean R2": -999,
                "CV Std R2": 0,
                "Model": None,
                "Error": str(e)
            }

    best_name = max(results, key=lambda k: results[k]["CV Mean R2"])
    return {
        "best_model_name": best_name,
        "best_model": results[best_name]["Model"],
        "all_results": results
    }