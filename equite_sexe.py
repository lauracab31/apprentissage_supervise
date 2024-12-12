from matplotlib import pyplot as plt
import pandas as pd
import joblib as jl
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


X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(hommes.drop('PINCP', axis=1)[:num_samples], hommes['PINCP'][:num_samples], test_size=0.2, random_state=42)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(femmes.drop('PINCP', axis=1)[:num_samples], femmes['PINCP'][:num_samples], test_size=0.2, random_state=42)


rf = jl.load('gridSearch_rf.joblib')
ab = jl.load('gridSearch_ab.joblib')
gb = jl.load('gridSearch_gb.joblib')

rf.fit(X_train_male, y_train_male)
ab.fit(X_train_male, y_train_male)
gb.fit(X_train_male, y_train_male)

y_pred_rf_male = rf.predict(X_test_male)
y_pred_ab_male = ab.predict(X_test_male)
y_pred_gb_male = gb.predict(X_test_male)

confusion_rf_male = confusion_matrix(y_test_male, y_pred_rf_male)
confusion_ab_male = confusion_matrix(y_test_male, y_pred_ab_male)
confusion_gb_male = confusion_matrix(y_test_male, y_pred_gb_male)

print("Matrice de confusion Random Forest (Homme):\n", confusion_rf_male)
print("Matrice de confusion Adaboost (Homme):\n", confusion_ab_male)
print("Matrice de confusion Gradient Boosting (Homme):\n", confusion_gb_male)

rf.fit(X_train_female, y_train_female)
ab.fit(X_train_female, y_train_female)
gb.fit(X_train_female, y_train_female)

y_pred_rf_female = rf.predict(X_test_female)
y_pred_ab_female = ab.predict(X_test_female)
y_pred_gb_female = gb.predict(X_test_female)

confusion_rf_female = confusion_matrix(y_test_female, y_pred_rf_female)
confusion_ab_female = confusion_matrix(y_test_female, y_pred_ab_female)
confusion_gb_female = confusion_matrix(y_test_female, y_pred_gb_female)

print("Matrice de confusion Random Forest (Femme):\n", confusion_rf_female)
print("Matrice de confusion Adaboost (Femme):\n", confusion_ab_female)
print("Matrice de confusion Gradient Boosting (Femme):\n", confusion_gb_female)


