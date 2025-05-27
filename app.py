from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Cargar modelos desde la carpeta "modelos"
modelos = {
    "MLP": joblib.load("modelos/modelo_mlp.pkl"),
    "XGBoost": joblib.load("modelos/modelo_xgb.pkl"),
    "LightGBM": joblib.load("modelos/modelo_lgb.pkl"),
    "HistGradientBoosting": joblib.load("modelos/modelo_hist.pkl")
}

niveles = {0: "Bajo", 1: "Medio", 2: "Alto"}

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

    # Agregar info personal y predicciones
    entrada_df["edad"] = edad
    entrada_df["genero"] = genero
    entrada_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for nombre in modelos:
        entrada_df[f"pred_{nombre}"] = resultados[nombre]

    archivo = "respuestas_pss10_multi.csv"
    if os.path.exists(archivo):
        entrada_df.to_csv(archivo, mode="a", index=False, header=False)
    else:
        entrada_df.to_csv(archivo, index=False)

    return jsonify(resultados)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)