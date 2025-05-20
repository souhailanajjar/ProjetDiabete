import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv("diabetes_cleaned.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Charger le scaler
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Liste des modèles à évaluer
models = {
    "Logistic Regression": joblib.load("LogisticRegression_model.pkl"),
    "K-Nearest Neighbors": joblib.load("KNN_model.pkl"),
    "Random Forest": joblib.load("RandomForest_model.pkl"),
    "SVM": joblib.load("SVM_model.pkl")
}

# Fonction d’évaluation
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n🔎 {name}")
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        print(f"AUC-ROC        : {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Évaluer tous les modèles
for name, model in models.items():
    evaluate_model(name, model)
