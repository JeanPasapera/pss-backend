import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

# Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# Dataset simulado
np.random.seed(42)
data = pd.DataFrame(np.random.randint(0, 5, size=(300, 10)), columns=[f"Q{i+1}" for i in range(10)])
data["score"] = data.sum(axis=1)
data["estrés"] = pd.cut(data["score"], bins=[-1,13,26,40], labels=["Bajo", "Medio", "Alto"])

X = data.drop(["score", "estrés"], axis=1)
y = LabelEncoder().fit_transform(data["estrés"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos = {
    "mlp": MLPClassifier(max_iter=1000, random_state=42),
    "hist": HistGradientBoostingClassifier(random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "lgb": LGBMClassifier(random_state=42)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, f"modelos/modelo_{nombre}.pkl")

print("✅ Modelos entrenados y guardados en /modelos/")