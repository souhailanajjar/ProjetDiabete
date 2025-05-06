import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Charger les données nettoyées
df = pd.read_csv("diabetes_cleaned.csv")

# Séparer les variables explicatives (X) et cible (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sauvegarder le scaler pour l'utiliser dans l'interface
joblib.dump(scaler, "scaler.pkl")

# Séparer en données d'entraînement et de test 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèles à entraîner
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)  # pour avoir accès à predict_proba
}

# Entraînement et sauvegarde
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")
    print(f"{name} entraîné et sauvegardé avec succès.")
