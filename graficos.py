import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
df = pd.DataFrame(list(client.pss_datos.respuestas.find()))

conteos = {}
for modelo in ["MLP", "XGBoost", "LightGBM", "HistGradientBoosting"]:
    conteos[modelo] = df[f"pred_{modelo}"].value_counts()

df_conteos = pd.DataFrame(conteos).fillna(0).astype(int)
df_conteos.plot(kind="bar")
plt.title("Distribución de niveles de estrés por modelo")
plt.ylabel("Cantidad")
plt.xlabel("Nivel de estrés")
plt.tight_layout()
plt.show()