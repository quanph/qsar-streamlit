# qsar3d_utils.py

"""
Module hỗ trợ mở rộng công cụ QSAR với:
✅ 3D-QSAR: chuẩn bị dữ liệu 3D, tính descriptor 3D (phần mềm ngoài: RDKit, Pybel, Mordred)
✅ AI nâng cao: Graph Neural Networks (GNN), ChemBERTa (hoá học + NLP)
"""

def prepare_3d_structures(smiles_list):
    """
    Chuyển SMILES thành cấu trúc 3D (mở rộng trong bước sau với RDKit + UFF)
    Returns danh sách Molecule 3D hoặc đường dẫn file SDF
    """
    raise NotImplementedError("🧪 Hàm chuẩn bị 3D sẽ được bổ sung với RDKit EmbedMultipleConfs()")

def calculate_3d_descriptors(sdf_path):
    """
    Tính toán 3D descriptors từ file .sdf (gợi ý dùng Mordred hoặc Pybel)
    """
    raise NotImplementedError("🔬 Hàm tính descriptor 3D sẽ sử dụng Mordred/Pybel hoặc phần mềm ngoài.")

def run_gnn_qsar(graph_dataset):
    """
    Huấn luyện mô hình Graph Neural Network từ dữ liệu phân tử dạng đồ thị
    Gợi ý dùng: DGL, PyTorch Geometric, DeepChem
    """
    raise NotImplementedError("🤖 Hàm huấn luyện GNN sẽ tích hợp GCN/GAT trong tương lai.")

def run_chemberta_prediction(smiles_list):
    """
    Dự đoán hoạt tính bằng ChemBERTa hoặc mô hình transformers cho SMILES
    """
    raise NotImplementedError("📚 Hàm tích hợp ChemBERTa cần mô hình đã fine-tuned.")