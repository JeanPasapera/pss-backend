import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

datos = joblib.load("datos_procesados.pkl")
label_encoder = joblib.load("label_encoder.pkl")

X_train, X_test = datos["X_train"], datos["X_test"]
y_train, y_test = datos["y_train"], datos["y_test"]

modelo = HistGradientBoostingClassifier(random_state=42)
modelo.fit(X_train, y_train)

y_pred_test = modelo.predict(X_test)
print(f"✅ HistGradientBoosting Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

joblib.dump(modelo, "modelos/modelo_hist.pkl")