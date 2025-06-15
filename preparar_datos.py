import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv("dataset_pss10_real.csv")

X = df[[f"Q{i+1}" for i in range(10)]]
y_str = df["estrés"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

joblib.dump({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}, "datos_procesados.pkl")

joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Datos preparados y guardados correctamente.")