import pandas as pd

# Charger la dataset
df = pd.read_csv("diabetes.csv")

# Afficher les 5 premières lignes
print("Aperçu des données :")
print(df.head())

# Résumé des infos
print("\nInfos générales :")
print(df.info())

# Statistiques descriptives
print("\nStatistiques :")
print(df.describe())

# Colonnes où 0 = valeur manquante
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Remplacer les 0 par NaN
df[cols_with_zero] = df[cols_with_zero].replace(0, pd.NA)

# Afficher les valeurs manquantes
print("\nValeurs manquantes :")
print(df.isna().sum())

# Remplacer les NaN par la médiane de chaque colonne
df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

# Supprimer les doublons
df_cleaned = df.drop_duplicates()

# Vérification finale
print("\nDonnées nettoyées :")
print(df_cleaned.info())

# Sauvegarde
df_cleaned.to_csv("diabetes_cleaned.csv", index=False)
print("\n✅ Dataset nettoyée enregistrée sous 'diabetes_cleaned.csv'")
