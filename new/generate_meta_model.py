import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from meta.meta_feature_extraction import extract_meta_features
from meta.base_models import MODELS

rows = []

datasets = [
    load_iris(),
    load_breast_cancer(),
]

for dataset in datasets:

    X = dataset.data
    y = dataset.target

    # Step 1: Extract meta features
    meta = extract_meta_features(X, y)

    # Step 2: Try all algorithms
    for name, model in MODELS.items():

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        row = meta.copy()
        row["algorithm"] = name
        row["accuracy"] = acc   # ✅ REAL performance

        rows.append(row)

# Step 3: Save dataset
df = pd.DataFrame(rows)
df.to_csv("meta_dataset.csv", index=False)

print("Meta dataset generated successfully")