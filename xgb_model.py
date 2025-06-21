import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

datos = joblib.load("datos_procesados.pkl")
X_train, X_test = datos["X_train"], datos["X_test"]
y_train, y_test = datos["y_train"], datos["y_test"]
label_encoder = joblib.load("label_encoder.pkl")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = XGBClassifier(
    n_estimators=100,                  
    max_depth=3,                        
    learning_rate=0.03,                 
    subsample=0.7,                      
    colsample_bytree=0.7,
    reg_alpha=0.1,                      
    reg_lambda=1.0,                     
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("🔧 XGBoost Metrics")
print(f"Precision:   {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall:      {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score:    {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

joblib.dump(modelo, "modelos/modelo_xgb.pkl")
joblib.dump(scaler, "modelos/scaler.pkl")