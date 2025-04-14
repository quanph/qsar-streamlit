import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import io
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt

st.set_page_config(page_title="QSAR Modeling Tool", layout="wide")
st.title("🔬 QSAR Modeling Tool – Streamlit App")

def calculate_descriptors(smiles_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        row = [desc[1](mol) if mol else np.nan for desc in Descriptors._descList]
        data.append(row)
    return pd.DataFrame(data, columns=desc_names)

# Upload and process
st.sidebar.header("📂 Upload QSAR Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    if "Atc" not in df.columns:
        st.warning("⚠️ Missing 'Atc' target column.")
    else:
        st.subheader("📊 Uploaded Data")
        st.dataframe(df.head())
        smiles = df["SMILES"] if "SMILES" in df.columns else None
        if smiles is not None:
            X_raw = calculate_descriptors(smiles)
        else:
            X_raw = df.drop(columns=["Atc"], errors="ignore")
        y = df["Atc"]

        # Config
        st.sidebar.header("⚙️ Model Config")
        use_kbest = st.sidebar.checkbox("Use SelectKBest", True)
        use_pca = st.sidebar.checkbox("Use PCA", False)
        test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
        run_automl = st.sidebar.checkbox("Use H2O AutoML")

        if st.sidebar.button("🚀 Train Models"):
            top_features = X_raw.corrwith(y).abs().sort_values(ascending=False).head(15).index.tolist()
            X_top = X_raw[top_features]
            if use_kbest:
                selector = SelectKBest(score_func=f_regression, k=min(10, len(top_features)))
                X_selected = selector.fit_transform(X_top, y)
            else:
                X_selected = X_top.values
            if use_pca:
                pca = PCA(n_components=5)
                X_selected = pca.fit_transform(X_selected)

            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "RandomForest": RandomForestRegressor(),
                "SVR": SVR(),
                "Ridge": Ridge(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "LightGBM": LGBMRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                results[name] = {
                    "Model": model,
                    "R2": r2,
                    "RMSE": rmse,
                    "CV Mean R2": cv.mean(),
                    "CV Std": cv.std()
                }

            best_model_name = max(results, key=lambda k: results[k]["CV Mean R2"])
            best_model = results[best_model_name]["Model"]
            st.success(f"🏆 Best model: {best_model_name} (CV R² = {results[best_model_name]['CV Mean R2']:.3f})")

            y_pred = best_model.predict(X_test)
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')
            ax.set_title(f"{best_model_name} Predictions")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            result_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "Model"} for k, v in results.items()}).T
            st.dataframe(result_df.style.format("{:.3f}"))

        if run_automl:
            st.subheader("🤖 H2O AutoML")
            h2o.init()
            df_h2o = h2o.H2OFrame(df)
            x_cols = [c for c in df.columns if c != "Atc"]
            aml = H2OAutoML(max_models=10, seed=1)
            aml.train(x=x_cols, y="Atc", training_frame=df_h2o)
            st.write(aml.leader)
            h2o.shutdown(prompt=False)