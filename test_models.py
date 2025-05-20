import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Préparer les données
@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("diabetes_cleaned.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Charger les modèles
@pytest.fixture(scope="module")
def models():
    return {
        "Logistic Regression": joblib.load("LogisticRegression_model.pkl"),
        "KNN": joblib.load("KNN_model.pkl"),
        "Random Forest": joblib.load("RandomForest_model.pkl"),
        "SVM": joblib.load("SVM_model.pkl")
    }

# Test 1 : Vérifier chargement des modèles
def test_models_loaded(models):
    for name, model in models.items():
        assert model is not None, f"{name} n'a pas été chargé."

# Test 2 : Vérifier que les modèles peuvent prédire
def test_models_predict(models, data):
    X_train, X_test, y_train, y_test = data
    for name, model in models.items():
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test), f"{name} n'a pas renvoyé le bon nombre de prédictions."

# Test 3 : Vérifier que les modèles atteignent une accuracy minimale (ex : 0.70)
def test_models_accuracy_threshold(models, data):
    X_train, X_test, y_train, y_test = data
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        assert acc >= 0.70, f"{name} a une accuracy trop basse : {acc:.2f}"

# Test 4 : Entrée vide
def test_model_with_empty_input(models):
    for name, model in models.items():
        with pytest.raises(ValueError):
            model.predict(np.array([]).reshape(1, -1))

# Test 5 : Mauvaise forme d'entrée (ex: mauvais nombre de colonnes)
def test_model_with_wrong_shape(models):
    bad_input = np.random.rand(1, 5)  # Doit avoir 8 colonnes
    for name, model in models.items():
        with pytest.raises(ValueError):
            model.predict(bad_input)

# Test 6 : Mauvais type d'entrée (ex: texte au lieu de float)
def test_model_with_wrong_type(models):
    wrong_input = np.array([["a", "b", "c", "d", "e", "f", "g", "h"]])
    for name, model in models.items():
        with pytest.raises(ValueError):
            model.predict(wrong_input)
