import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = joblib.load("datos_procesados.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
label_encoder = joblib.load("label_encoder.pkl")

modelo = MLPClassifier(max_iter=300, random_state=42)
print("🔧 Entrenando modelo MLP...")
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acc_train = accuracy_score(y_train, modelo.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print(f"✅ Precisión entrenamiento: {acc_train:.4f}")
print(f"✅ Precisión prueba:       {acc_test:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(modelo, "modelos/modelo_mlp.pkl")
print("💾 Modelo MLP guardado en 'modelos/modelo_mlp.pkl'")