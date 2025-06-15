import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("modelos", exist_ok=True)

data_path = "dataset_pss10_real.csv"
df = pd.read_csv(data_path)

X = df[[f"Q{i+1}" for i in range(10)]]
y_str = df["estrés"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

modelos = {
    "mlp": MLPClassifier(max_iter=300, random_state=42),
    "hist": HistGradientBoostingClassifier(random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "lgb": LGBMClassifier(
        random_state=42,
        num_leaves=31,
        min_data_in_leaf=1,
        max_depth=5,
        verbose=-1
    )
}

for nombre, modelo in modelos.items():
    print(f"\n🔧 Entrenando modelo: {nombre.upper()}...")
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, f"modelos/modelo_{nombre}.pkl")

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test = accuracy_score(y_test, modelo.predict(X_test))

    print(f"   - Precisión entrenamiento: {acc_train:.4f}")
    print(f"   - Precisión prueba:       {acc_test:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, modelo.predict(X_test), target_names=label_encoder.classes_))

print("\n✅ Modelos entrenados correctamente y guardados en /modelos/")