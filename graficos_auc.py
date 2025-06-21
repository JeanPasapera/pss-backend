import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

client = MongoClient("mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
df = pd.DataFrame(list(client.pss_datos.respuestas.find()))

df = df.dropna(subset=["respuestas", "pred_MLP"])
df = df[df["pred_MLP"].isin(["Bajo", "Medio", "Alto"])]

X = pd.DataFrame(df["respuestas"].tolist(), columns=[f"Q{i+1}" for i in range(10)])
y_raw = df["pred_MLP"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

modelos = {
    "MLP": MLPClassifier(max_iter=300, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

for nombre, modelo in modelos.items():
    print(f"🔧 Entrenando: {nombre}")
    modelo.fit(X_train, y_train)
    y_score = modelo.predict_proba(X_test)

    plt.figure(figsize=(7, 5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
        auc_score = roc_auc_score(y_test[:, i], y_score[:, i])
        label = f"{label_encoder.inverse_transform([i])[0]} (AUC = {auc_score:.2f})"
        plt.plot(fpr, tpr, lw=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f"AUC-ROC Curve - {nombre}", fontsize=13)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()