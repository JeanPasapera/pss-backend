import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

client = MongoClient("mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
df = pd.DataFrame(list(client.pss_datos.respuestas.find()))

df = df.dropna(subset=["respuestas", "pred_MLP"])
df = df[df["pred_MLP"].isin(["Bajo", "Medio", "Alto"])]

X = pd.DataFrame(df["respuestas"].tolist(), columns=[f"Q{i+1}" for i in range(10)])
y = LabelEncoder().fit_transform(df["pred_MLP"])

modelos = {
    "MLP": MLPClassifier(max_iter=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

for nombre, modelo in modelos.items():
    acc_train_list = []
    acc_test_list = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        modelo.fit(X_train, y_train)

        acc_train = accuracy_score(y_train, modelo.predict(X_train))
        acc_test = accuracy_score(y_test, modelo.predict(X_test))

        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

    plt.figure(figsize=(8, 5))
    plt.plot(acc_train_list, label="Train Accuracy", linestyle="--", marker='o')
    plt.plot(acc_test_list, label="Validation Accuracy", marker='s')
    plt.title(f"{nombre} Accuracy Over Iterations", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(range(10))
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()