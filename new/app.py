import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from meta.meta_feature_extraction import extract_meta_features
from meta.base_models import MODELS

st.set_page_config(page_title="Meta AutoML System", layout="centered")

st.title("Meta-Learning Based AutoML System")
st.caption("Upload a dataset to predict best ML algorithm")

# Load trained meta-model
meta_model = joblib.load("artifacts/meta_model.pkl")
meta_columns = joblib.load("artifacts/meta_columns.pkl")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset (Last column = target)", type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Analyze Dataset"):

        # ---------------- DATA SPLIT ----------------
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]



        # ---------------- DATA CLEANING ----------------
        # Convert categorical to numeric
        X = pd.get_dummies(X)
        # Remove ID column if exists
        if "id" in X.columns:
            X = X.drop(columns=["id"])
        # Ensure numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Fill missing values
        X = X.fillna(0)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert target if categorical
        
        y = pd.factorize(y)[0]
        y = y.astype(int)
        y = y.ravel()

        # ---------------- META FEATURES ----------------
        meta_features = extract_meta_features(X, y)

        # ---------------- TRAIN-TEST SPLIT ----------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        results = []

        # ---------------- MODEL LOOP ----------------
        for algo_name, model in MODELS.items():

            # ===== META MODEL PREDICTION =====
            row = meta_features.copy()
            row["algorithm"] = algo_name

            meta_input = pd.DataFrame([row])
            meta_input = pd.get_dummies(meta_input)
            meta_input = meta_input.reindex(columns=meta_columns, fill_value=0)

            predicted_acc = meta_model.predict(meta_input)[0]

            # ===== REAL MODEL TRAINING =====


            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            acc = scores.mean()

# Train once for other metrics
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results.append({
                "Algorithm": algo_name,
                "Meta Predicted Accuracy": round(predicted_acc, 3),
                "Actual Accuracy": round(acc, 3),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1 Score": round(f1, 3)
            })

        # ---------------- RESULTS ----------------
        results_df = pd.DataFrame(results).sort_values(
            by="Actual Accuracy", ascending=False
        )
        st.write("Predictions sample:", preds[:10])
        st.write("Actual sample:", y_test[:10])

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)

        # ---------------- BEST MODEL ----------------
        best = results_df.iloc[0]

        st.success(f"Best Algorithm: {best['Algorithm']}")