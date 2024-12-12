import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib as jl

# Charger les données
df = pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

# Standardisation des attributs numériques
scalerX = StandardScaler()
features_scaled = scalerX.fit_transform(df)
jl.dump(scalerX, "scaler.jl")

# Analyse des corrélations initiales
chi_scores, p_values = chi2(df, lbl)
print("chi scores = ", chi_scores)
print("p values = ", p_values)

df["PINCP"] = lbl  # Ajouter la colonne cible (revenu)
correlation_initial = df.corr()
print("Matrice de corrélation initiale :")
print(correlation_initial)

# Visualisation de la matrice de corrélation initiale
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_initial, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Matrice de corrélation initiale")
plt.show()


# Standardisation des attributs numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
y = lbl.values.ravel()

# Partitionnement des données
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X_scaled, y))
X_train, X_test = X_scaled[train_index], X_scaled[test_index]
y_train, y_test = y[train_index], y[test_index]

# Entraîner un modèle Random Forest
rf_model = jl.load('gridSearch_rf.joblib')
rf_model.fit(X_train, y_train)

# Ajouter les prédictions comme nouvelle colonne
train_predictions = rf_model.predict(X_train)
train_df = pd.DataFrame(X_train, columns=df.columns)  

# Calculer la matrice de corrélation
correlation_rf = train_df.corr()

# Afficher la matrice de corrélation
print("Matrice de corrélation après entraînement avec RandomForest :")
print(correlation_rf)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_rf,annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Matrice de corrélation - RandomForest avec prédictions")
plt.show()

# Charger le modèle AdaBoost préalablement enregistré
ab_model = jl.load('gridSearch_ab.joblib')

# Assurez-vous que X_train contient exactement les mêmes colonnes que celles utilisées pour entraîner le modèle
# Par exemple, si le modèle a été entraîné avec 10 colonnes spécifiques, assurez-vous que ces mêmes colonnes sont présentes dans X_train.

# Option 1: Si vous avez une liste des colonnes exactes utilisées pendant l'entraînement :
expected_columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']
X_train = X_train[expected_columns]  # Garder uniquement les colonnes attendues

# Effectuer les prédictions
train_predictions = ab_model.predict(X_train)

# Ajouter les prédictions comme nouvelle colonne dans le DataFrame
train_df = pd.DataFrame(X_train, columns=expected_columns)
train_df['predictions'] = train_predictions

# Calculer la matrice de corrélation
correlation_ab = train_df.corr()

# Afficher la matrice de corrélation
print("Matrice de corrélation après entraînement avec AdaBoost :")
print(correlation_ab)

# Visualiser la matrice de corrélation avec un heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_ab, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Matrice de corrélation - AdaBoost avec prédictions")
plt.show()


