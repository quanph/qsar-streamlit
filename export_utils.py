import pandas as pd

def export_to_csv(df, filename='qsar_predictions.csv'):
    """
    Xuất DataFrame ra file CSV

    Args:
        df (pd.DataFrame): Dữ liệu cần xuất
        filename (str): Tên file CSV

    Returns:
        str: Đường dẫn file đã lưu
    """
    df.to_csv(filename, index=False)
    return filename

def export_to_excel(df, filename='qsar_predictions.xlsx'):
    """
    Xuất DataFrame ra file Excel

    Args:
        df (pd.DataFrame): Dữ liệu cần xuất
        filename (str): Tên file Excel

    Returns:
        str: Đường dẫn file đã lưu
    """
    df.to_excel(filename, index=False)
    return filename