import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler



import joblib as jl

#extraire features dans une dataframe et la visualiser
df= pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

#print (df)

# Standardisation des attributs numériques: centrer et réduire les données (moyenne = 0, écart-type = 1)pour les algorithmes sensibles à l'échelle des données comme les SVM ou KNN.
scalerX=StandardScaler()
scaler2=scalerX.fit(df)
features_scaled=scaler2.transform(df)
#enregistrement des valeurs du scalerX
jl.dump(scalerX,"scaler.jl")

#Analyse des corrélations initiales avec le test chi2
chi_scores, p_values = chi2(df, lbl)
print("chi scores = ", chi_scores)
print("p values = ", p_values)

df["PINCP"]=lbl
corelation_matrix_initiale = df.corr()

print("Matrice de corrélation initiale :")
print(corelation_matrix_initiale)

# Visualisation de la matrice de corrélation initiale
plt.figure(figsize=(12, 8))
sns.heatmap(corelation_matrix_initiale, annot=False, cmap="coolwarm")
plt.title("Matrice de corrélation initiale")
plt.show()

# Partitionnement des données
X = features_scaled
y = lbl.values.ravel()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Obtenir les prédictions
y_pred_train = rf_model.predict(X_train)

# Créer des DataFrames pour analyser les corrélations avec les prédictions
train_df = pd.DataFrame(X_train, columns=df.columns[:-1])  # Exclure la colonne 'PINCP'
train_df['prediction'] = y_pred_train
train_df['label'] = y_train

# Corrélations après apprentissage (train et test)
correlation_train = train_df.corr()

print("Matrice de corrélation sur l'ensemble d'entraînement (avec prédictions) :")
print(correlation_train)

# Visualisation des matrices de corrélation avec prédictions
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_train, annot=False, cmap="coolwarm")
plt.title("Matrice de corrélation avec Random Forest")
plt.show()