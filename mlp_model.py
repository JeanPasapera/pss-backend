import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

data = joblib.load("datos_procesados.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
label_encoder = joblib.load("label_encoder.pkl")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,             
    n_iter_no_change=10,
    random_state=42
)

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("🔧 MLP Metrics")
print(f"Precision:   {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall:      {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(modelo, "modelos/modelo_mlp.pkl")
joblib.dump(scaler, "modelos/scaler.pkl")