import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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


#étudier la distribution des features
featuresEtudiees = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
df.describe()
#for colonne in featuresEtudiees :
#    plt.hist(df[colonne], bins=50, alpha=0.5, label=colonne)
#    plt.xlabel('features')
#    plt.ylabel('valeurs')
#    plt.title('distribution des features')
#    plt.legend()
   # plt.show()
   # plt.savefig('hist'+colonne, dpi=300)

# Préparation des données pour le partitionnement
X = features_scaled
y = lbl.values.ravel()  # Convertir en format numpy (1D) compatible avec StratifiedShuffleSplit

# Partitionnement du jeu de données
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Taille train:", X_train.shape, "Taille test:", X_test.shape)

# Vérification des dimensions après traitement
print(f"\nDimensions des données d'entrée X : {X.shape}")
print(f"Dimensions des labels y : {y.shape}")

#recherche de bons modèles
#on commence avec le modèle type RandomForest :
rf_model = RandomForestClassifier()
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
print("Moyenne des scores de Validation croisée = ", cv_scores_rf)

#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
