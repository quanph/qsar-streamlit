import joblib
import pandas as pd

def predict_from_descriptors(descriptor_df, model_path, scaler_path, selector_path=None, pca_path=None):
    """
    Dự đoán từ bảng descriptor với pipeline đã lưu

    Args:
        descriptor_df (pd.DataFrame): Bảng descriptor đã tính từ SMILES
        model_path (str): Đường dẫn file mô hình .pkl
        scaler_path (str): Đường dẫn scaler .pkl
        selector_path (str, optional): Đường dẫn selector .pkl (SelectKBest)
        pca_path (str, optional): Đường dẫn PCA .pkl

    Returns:
        np.ndarray: Vector giá trị dự đoán
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X = descriptor_df.copy()

    if selector_path:
        selector = joblib.load(selector_path)
        X = selector.transform(X)

    if pca_path:
        pca = joblib.load(pca_path)
        X = pca.transform(X)

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions