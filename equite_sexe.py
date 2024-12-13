from matplotlib import pyplot as plt
import pandas as pd
import joblib as jl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

df = pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

data = pd.concat([df, lbl], axis=1)
num_samples = int(len(data) * 0.1)

hommes = data[data['SEX'] == 1]
femmes = data[data['SEX'] == 2]

taux_total = data['PINCP'].mean() * 100
print(f"Taux global d'individus ayant un revenu supérieur à 50 000 dollars: {taux_total:.2f}%")



taux_hommes = hommes['PINCP'].mean() * 100
print(f"Taux d'hommes ayant un revenu supérieur à 50 000 dollars: {taux_hommes:.2f}%")

taux_femmes = femmes['PINCP'].mean() * 100
print(f"Taux de femmes ayant un revenu supérieur à 50 000 dollars: {taux_femmes:.2f}%")


X_train_homme, X_test_homme, y_train_homme, y_test_homme = train_test_split(hommes.drop('PINCP', axis=1)[:num_samples], hommes['PINCP'][:num_samples], test_size=0.2, random_state=42)
X_train_femme, X_test_femme, y_train_femme, y_test_femme = train_test_split(femmes.drop('PINCP', axis=1)[:num_samples], femmes['PINCP'][:num_samples], test_size=0.2, random_state=42)


rf = jl.load('gridSearch_rf.joblib')
ab = jl.load('gridSearch_ab.joblib')
gb = jl.load('gridSearch_gb.joblib')

rf.fit(X_train_homme, y_train_homme)
ab.fit(X_train_homme, y_train_homme)
gb.fit(X_train_homme, y_train_homme)

y_pred_rf_homme = rf.predict(X_test_homme)
y_pred_ab_homme = ab.predict(X_test_homme)
y_pred_gb_homme = gb.predict(X_test_homme)

confusion_rf_homme = confusion_matrix(y_test_homme, y_pred_rf_homme)
confusion_ab_homme = confusion_matrix(y_test_homme, y_pred_ab_homme)
confusion_gb_homme = confusion_matrix(y_test_homme, y_pred_gb_homme)

print("Matrice de confusion Random Forest pour les hommes:\n", confusion_rf_homme)
print("Matrice de confusion Adaboost pour les hommes:\n", confusion_ab_homme)
print("Matrice de confusion Gradient Boosting pour les hommes:\n", confusion_gb_homme)

rf.fit(X_train_femme, y_train_femme)
ab.fit(X_train_femme, y_train_femme)
gb.fit(X_train_femme, y_train_femme)

y_pred_rf_femme = rf.predict(X_test_femme)
y_pred_ab_femme = ab.predict(X_test_femme)
y_pred_gb_femme = gb.predict(X_test_femme)

confusion_rf_femme = confusion_matrix(y_test_femme, y_pred_rf_femme)
confusion_ab_femme = confusion_matrix(y_test_femme, y_pred_ab_femme)
confusion_gb_femme = confusion_matrix(y_test_femme, y_pred_gb_femme)

print("Matrice de confusion Random Forest pour les femmes:\n", confusion_rf_femme)
print("Matrice de confusion Adaboost pour les femmes:\n", confusion_ab_femme)
print("Matrice de confusion Gradient Boosting pour les femmes:\n", confusion_gb_femme)

#Equité des modèles : refaire un entrainement sur le jeu de données dans lequel on a retiré la colonne de genre
# Deux instances distinctes du modèle Random Forest
rf_homme = jl.load('gridSearch_rf.joblib')
rf_femme = jl.load('gridSearch_rf.joblib')

train_sans_sex_homme = X_train_homme.drop(columns=['SEX'])
train_sans_sex_femme = X_train_femme.drop(columns=['SEX'])

test_sans_sex_homme = X_test_homme.drop(columns=['SEX'])
test_sans_sex_femme = X_test_femme.drop(columns=['SEX'])

# Entraînement sur les données sans la colonne SEX
rf_homme.fit(train_sans_sex_homme, y_train_homme)
rf_femme.fit(train_sans_sex_femme, y_train_femme)

y_pred_rf_sansSex_homme = rf_homme.predict(test_sans_sex_homme)
y_pred_rf_sansSex_femme = rf_femme.predict(test_sans_sex_femme)

# Matrices de confusion
confusion_rf_sansSex_homme = confusion_matrix(y_test_homme, y_pred_rf_sansSex_homme)
confusion_rf_sansSex_femme = confusion_matrix(y_test_femme, y_pred_rf_sansSex_femme)

# Affichage des matrices de confusion
print("Matrice de confusion sans colonne de genre en Random Forest pour les hommes:\n", confusion_rf_sansSex_homme)
print("Matrice de confusion sans colonne de genre en Random Forest pour les femmes:\n", confusion_rf_sansSex_femme)

ab_homme = jl.load('gridSearch_ab.joblib')
ab_femme = jl.load('gridSearch_ab.joblib')

ab_homme.fit(train_sans_sex_homme, y_train_homme)
ab_femme.fit(train_sans_sex_femme, y_train_femme)

y_pred_ab_sansSex_homme = ab_homme.predict(test_sans_sex_homme)
y_pred_ab_sansSex_femme = ab_femme.predict(test_sans_sex_femme)

# Matrices de confusion
confusion_ab_sansSex_homme = confusion_matrix(y_test_homme, y_pred_ab_sansSex_homme)
confusion_ab_sansSex_femme = confusion_matrix(y_test_femme, y_pred_ab_sansSex_femme)

# Affichage des matrices de confusion
print("Matrice de confusion sans colonne de genre en AdaBoost pour les hommes:\n", confusion_ab_sansSex_homme)
print("Matrice de confusion sans colonne de genre en AdaBoost pour les femmes:\n", confusion_ab_sansSex_femme)

gb_homme = jl.load('gridSearch_gb.joblib')
gb_femme = jl.load('gridSearch_gb.joblib')

gb_homme.fit(train_sans_sex_homme, y_train_homme)
gb_femme.fit(train_sans_sex_femme, y_train_femme)

y_pred_gb_sansSex_homme = gb_homme.predict(test_sans_sex_homme)
y_pred_gb_sansSex_femme = gb_femme.predict(test_sans_sex_femme)

# Matrices de confusion
confusion_gb_sansSex_homme = confusion_matrix(y_test_homme, y_pred_gb_sansSex_homme)
confusion_gb_sansSex_femme = confusion_matrix(y_test_femme, y_pred_gb_sansSex_femme)

# Affichage des matrices de confusion
print("Matrice de confusion sans colonne de genre en GradientBoosting pour les hommes:\n", confusion_gb_sansSex_homme)
print("Matrice de confusion sans colonne de genre en GradientBoosting pour les femmes:\n", confusion_gb_sansSex_femme)