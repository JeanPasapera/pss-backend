from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app, origins=["https://testpss10.christianovalle.com"])

modelos = {
    "MLP": joblib.load("modelos/modelo_mlp.pkl"),
    "XGBoost": joblib.load("modelos/modelo_xgb.pkl"),
    "LightGBM": joblib.load("modelos/modelo_lgb.pkl"),
    "HistGradientBoosting": joblib.load("modelos/modelo_hist.pkl")
}

niveles = {0: "Bajo", 1: "Medio", 2: "Alto"}

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://pss_user:Pasapera2310.@gptcluster.hj5l4pa.mongodb.net/?retryWrites=true&w=majority&appName=GPTCluster")
client = MongoClient(MONGO_URI)
db = client.pss_datos
coleccion = db.respuestas

@app.route("/")
def index():
    return "✅ API del Test PSS-10 con múltiples modelos está activa."

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        data = request.get_json()

        respuestas = data.get("respuestas", [])
        edad = data.get("edad", None)
        genero = data.get("genero", "Sin especificar")

        if not isinstance(respuestas, list) or len(respuestas) != 10:
            return jsonify({"error": "Debes enviar exactamente 10 respuestas."}), 400

        entrada_df = pd.DataFrame([respuestas], columns=[f"Q{i+1}" for i in range(10)])

        resultados = {}
        for nombre, modelo in modelos.items():
            pred = modelo.predict(entrada_df)[0]
            resultados[nombre] = niveles.get(pred, "Desconocido")

        registro = {
            "respuestas": respuestas,
            "edad": edad,
            "genero": genero,
            "timestamp": datetime.utcnow(),
        }

        for nombre in modelos:
            registro[f"pred_{nombre}"] = resultados[nombre]

        coleccion.insert_one(registro)

        return jsonify(resultados)

    except Exception as e:
        print("❌ Error en /predecir:", str(e))
        return jsonify({"error": "Error interno del servidor."}), 500

@app.route("/estadisticas", methods=["GET"])
def estadisticas():
    try:
        stats = {
            "MLP": {"Bajo": 0, "Medio": 0, "Alto": 0},
            "XGBoost": {"Bajo": 0, "Medio": 0, "Alto": 0},
            "LightGBM": {"Bajo": 0, "Medio": 0, "Alto": 0},
            "HistGradientBoosting": {"Bajo": 0, "Medio": 0, "Alto": 0},
        }

        for doc in coleccion.find():
            for modelo in stats.keys():
                pred = doc.get(f"pred_{modelo}")
                if pred in stats[modelo]:
                    stats[modelo][pred] += 1

        return jsonify(stats)

    except Exception as e:
        print("❌ Error en /estadisticas:", str(e))
        return jsonify({"error": "No se pudieron calcular las estadísticas."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)