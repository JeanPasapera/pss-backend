from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import os

app = Flask(__name__)
CORS(app)

modelos = {
    "MLP": joblib.load("modelos/modelo_mlp.pkl"),
    "XGBoost": joblib.load("modelos/modelo_xgb.pkl"),
    "LightGBM": joblib.load("modelos/modelo_lgb.pkl"),
    "HistGradientBoosting": joblib.load("modelos/modelo_hist.pkl")
}

niveles = {0: "Bajo", 1: "Medio", 2: "Alto"}

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://<pss_user>:<Pasapera2310.>@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
client = MongoClient(MONGO_URI)
db = client.pss
coleccion = db.respuestas

@app.route("/")
def index():
    return "✅ API del Test PSS-10 con múltiples modelos está activa."

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    respuestas = data.get("respuestas", [])
    edad = data.get("edad")
    genero = data.get("genero")

    if len(respuestas) != 10:
        return jsonify({"error": "Debes enviar exactamente 10 respuestas"}), 400

    entrada_df = pd.DataFrame([respuestas], columns=[f"Q{i+1}" for i in range(10)])
    resultados = {}

    for nombre, modelo in modelos.items():
        pred = modelo.predict(entrada_df)[0]
        resultados[nombre] = niveles.get(pred, "Desconocido")

    doc = {
        "respuestas": respuestas,
        "edad": edad,
        "genero": genero,
        "timestamp": datetime.now(),
        **{f"pred_{nombre}": resultados[nombre] for nombre in modelos}
    }

    coleccion.insert_one(doc)

    return jsonify(resultados)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

@app.route("/estadisticas", methods=["GET"])
def estadisticas():
    # Diccionario para almacenar resultados por modelo
    stats = {
        "MLP": {"Bajo": 0, "Medio": 0, "Alto": 0},
        "XGBoost": {"Bajo": 0, "Medio": 0, "Alto": 0},
        "LightGBM": {"Bajo": 0, "Medio": 0, "Alto": 0},
        "HistGradientBoosting": {"Bajo": 0, "Medio": 0, "Alto": 0},
    }

    # Recorrer todos los documentos
    for doc in coleccion.find():
        for modelo in stats.keys():
            pred = doc.get(f"pred_{modelo}")
            if pred in stats[modelo]:
                stats[modelo][pred] += 1

    return jsonify(stats)