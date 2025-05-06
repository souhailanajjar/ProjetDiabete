import streamlit as st
import pandas as pd
import joblib

# Titre de l'application
st.title("🩺 Classification du Diabète")
st.markdown("Sélectionnez un modèle de machine learning et entrez les informations du patient pour prédire le diabète (0 = Non, 1 = Oui).")

# Sélection du modèle
model_choice = st.selectbox("🧠 Choisissez un modèle :", ["RandomForest", "LogisticRegression", "KNN", "SVM"])

# Chargement du modèle et du scaler
try:
    model = joblib.load(f"{model_choice}_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Modèle ou scaler introuvable. Veuillez exécuter 'train_models.py' pour les générer.")
    st.stop()

# Saisie des données du patient avec précision sans arrondi
st.subheader("📥 Saisir les données du patient :")
Pregnancies = st.number_input("Nombre de grossesses", min_value=0)
Glucose = st.number_input("Taux de glucose", min_value=0.0, format="%.5f")
BloodPressure = st.number_input("Pression artérielle", min_value=0.0, format="%.5f")
SkinThickness = st.number_input("Épaisseur de peau", min_value=0.0, format="%.5f")
Insulin = st.number_input("Taux d'insuline", min_value=0.0, format="%.5f")
BMI = st.number_input("Indice de masse corporelle (BMI)", min_value=0.0, format="%.5f")
DiabetesPedigreeFunction = st.number_input("Antécédents familiaux (DPF)", min_value=0.0, format="%.5f")
Age = st.number_input("Âge", min_value=1)

# Bouton de prédiction
if st.button("🔮 Prédire"):
    # Vérification des champs (sauf Pregnancies qui peut être 0)
    if (Glucose == 0.0 or BloodPressure == 0.0 or SkinThickness == 0.0 or
        Insulin == 0.0 or BMI == 0.0 or DiabetesPedigreeFunction == 0.0 or Age == 1):
        st.error("❌ Veuillez remplir tous les champs obligatoires avant de prédire.")
    else:
        # Construction du DataFrame
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DiabetesPedigreeFunction, Age]],
                                  columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        # Normalisation des données
        input_scaled = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(input_scaled)[0]

        st.subheader("📊 Résultat :")
        st.write(f"👉 **{int(prediction)}** (0 = Non diabétique, 1 = Diabétique)")
