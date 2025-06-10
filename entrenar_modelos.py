import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("modelos", exist_ok=True)

client = MongoClient("mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
db = client.pss_datos
coleccion = db.respuestas

docs = list(coleccion.find())
df = pd.DataFrame(docs)

df = df.dropna(subset=["respuestas", "pred_MLP"])
df = df[df["pred_MLP"].isin(["Bajo", "Medio", "Alto"])]

X = pd.DataFrame(df["respuestas"].tolist(), columns=[f"Q{i+1}" for i in range(10)])
y = LabelEncoder().fit_transform(df["pred_MLP"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos = {
    "mlp": MLPClassifier(max_iter=150, random_state=42),
    "hist": HistGradientBoostingClassifier(random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "lgb": LGBMClassifier(random_state=42)
}

for nombre, modelo in modelos.items():
    print(f"Entrenando modelo: {nombre.upper()}...")
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, f"modelos/modelo_{nombre}.pkl")

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test = accuracy_score(y_test, modelo.predict(X_test))
    print(f"  - Precisión entrenamiento: {acc_train:.4f}")
    print(f"  - Precisión prueba: {acc_test:.4f}")

print("✅ Modelos entrenados y guardados en /modelos/")