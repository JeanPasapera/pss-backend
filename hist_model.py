import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

datos = joblib.load("datos_procesados.pkl")
X_train, X_test = datos["X_train"], datos["X_test"]
y_train, y_test = datos["y_train"], datos["y_test"]
label_encoder = joblib.load("label_encoder.pkl")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.03,
    max_leaf_nodes=25,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    random_state=42
)

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("🔧 HistGradientBoosting Metrics")
print(f"Precision:   {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall:      {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(modelo, "modelos/modelo_hist.pkl")
joblib.dump(scaler, "modelos/scaler.pkl")