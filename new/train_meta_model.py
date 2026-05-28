import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("meta_dataset.csv")

X = df.drop("accuracy", axis=1)
y = df["accuracy"]

X = pd.get_dummies(X)

meta_columns = X.columns

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump(model, "artifacts/meta_model.pkl")
joblib.dump(meta_columns, "artifacts/meta_columns.pkl")

print("Meta model trained (REGRESSION)")