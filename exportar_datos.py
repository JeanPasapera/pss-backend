import pandas as pd
from pymongo import MongoClient
import os

client = MongoClient("mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
db = client.pss_datos
coleccion = db.respuestas

docs = list(coleccion.find({
    "respuestas": {"$exists": True, "$type": "array", "$size": 10},
    "pred_MLP": {"$in": ["Bajo", "Medio", "Alto"]}
}))

df = pd.DataFrame(docs)
df_final = pd.DataFrame(df["respuestas"].tolist(), columns=[f"Q{i+1}" for i in range(10)])
df_final["estrés"] = df["pred_MLP"]
df_final["edad"] = df["edad"]
df_final["genero"] = df["genero"]
df_final["timestamp"] = df["timestamp"]

output = "dataset_pss10_real.csv"
df_final.to_csv(output, index=False, encoding="utf-8")

print(f"✅ Dataset exportado a {output} con {len(df_final)} registros.")