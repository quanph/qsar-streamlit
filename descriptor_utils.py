from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def calculate_basic_descriptors(smiles_list):
    """
    Tính toán toàn bộ molecular descriptors cơ bản từ RDKit
    Args:
        smiles_list (list): Danh sách SMILES string
    Returns:
        pd.DataFrame: Bảng các descriptor
    """
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        row = [desc[1](mol) if mol else None for desc in Descriptors._descList]
        data.append(row)
    return pd.DataFrame(data, columns=descriptor_names)