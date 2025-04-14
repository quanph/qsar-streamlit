import h2o
from h2o.automl import H2OAutoML
import pandas as pd

def run_h2o_automl(df, target_col='Atc', max_models=10, seed=1):
    """
    Chạy H2O AutoML trên DataFrame Pandas

    Args:
        df (pd.DataFrame): Dữ liệu đầu vào gồm cả đặc trưng và biến mục tiêu
        target_col (str): Tên cột mục tiêu
        max_models (int): Số lượng mô hình tối đa
        seed (int): Hạt giống ngẫu nhiên

    Returns:
        dict: thông tin mô hình tốt nhất và leaderboard
    """
    h2o.init()
    df_h2o = h2o.H2OFrame(df)
    x_cols = [col for col in df.columns if col != target_col]
    
    aml = H2OAutoML(max_models=max_models, seed=seed)
    aml.train(x=x_cols, y=target_col, training_frame=df_h2o)
    
    leader = aml.leader
    leaderboard = aml.leaderboard.as_data_frame()
    
    h2o.shutdown(prompt=False)
    
    return {
        "leader_model": leader,
        "leaderboard": leaderboard
    }