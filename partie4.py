import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import joblib as jl

#extraire features dans une dataframe et la visualiser
df= pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

data = pd.concat([df, lbl], axis=1)

# Échantillonnage : prendre 10% des données de manière aléatoire
data_sampled = data.sample(frac=0.1, random_state=42)

# Séparation en données d'entraînement et de test à partir du jeu de données complet
X_all = data.drop('PINCP', axis=1)
y_all = data['PINCP']
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Séparation en données d'entraînement et de test à partir du jeu de données échantillonné
X_sampled = data_sampled.drop('PINCP', axis=1)
y_sampled = data_sampled['PINCP']
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

# Fusion des données d'entraînement complètes (X_train_all et y_train_all)
train_all_df = pd.merge(X_train_all, y_train_all, left_index=True, right_index=True)

# Calcul de la matrice de corrélation pour les données d'entraînement complètes
correlation_matrix = train_all_df.corr()

# Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation :")
print(correlation_matrix)

# Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation des données initiales')
plt.show()

rf = jl.load('gridSearch_rf.joblib')

plt.figure(figsize=(10, 6))
correlation_matrix.iloc[:-1]['PINCP'].plot(kind='barh', color='red')
plt.title('Correlation with PINCP Random Forest')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

rf.fit(X_train, y_train)  # Entraînement du modèle

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_rf = rf.predict(X_test_all)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_rf = pd.DataFrame(y_pred_rf, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
merged_rf = pd.concat([X_test_all, df_y_pred_rf], axis=1)

# 5. Calcul de la matrice de corrélation
correlation_matrix_rf = merged_rf.corr()

# 6. Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation après entraînement avec RandomForest :")
print(correlation_matrix_rf)

# 7. Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_rf, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec RandomForest')
plt.show()

importance_rf = permutation_importance(rf, X_test_all, df_y_pred_rf, random_state=42).importances_mean

for i in range(len(importance_rf)):
    print(f"Feature {i}: ", importance_rf[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_rf, color='skyblue')
plt.title('Feature Importance Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

ab = jl.load('gridSearch_ab.joblib')

ab.fit(X_train, y_train)  # Entraînement du modèle

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_ab = ab.predict(X_test_all)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_ab = pd.DataFrame(y_pred_ab, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
merged_ab = pd.concat([X_test_all, df_y_pred_ab], axis=1)

# 5. Calcul de la matrice de corrélation
correlation_matrix_ab = merged_ab.corr()

# 6. Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation après entraînement avec AdaBoost :")
print(correlation_matrix_ab)

# 7. Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_ab, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec AdaBoost')
plt.show()

importance_ab = permutation_importance(ab, X_test_all, df_y_pred_ab, random_state=42).importances_mean

for i in range(len(importance_ab)):
    print(f"Feature {i}: ", importance_ab[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_ab, color='red')
plt.title('Feature Importance AdaBoost')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

gb = jl.load('gridSearch_gb.joblib')

gb.fit(X_train, y_train)  # Entraînement du modèle

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_gb = gb.predict(X_test_all)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_gb = pd.DataFrame(y_pred_gb, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
merged_gb = pd.concat([X_test_all, df_y_pred_gb], axis=1)

# 5. Calcul de la matrice de corrélation
correlation_matrix_gb = merged_gb.corr()

# 6. Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation après entraînement avec GB :")
print(correlation_matrix_gb)

# 7. Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_gb, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec GB')
plt.show()

importance_gb = permutation_importance(gb, X_test_all, df_y_pred_gb, random_state=42).importances_mean

for i in range(len(importance_gb)):
    print(f"Feature {i}: ", importance_gb[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_gb, color='green')
plt.title('Feature Importance GB')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

